import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


class TPULLMEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading {model_id} onto TPU...")

        self.device = xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        # Warm up the TPU so first real query isn't slow
        self._warmup()

        self.max_new_tokens = 150    # Keep short for Streamlit timeout budget
        self.max_input_tokens = 768

        print("TPU Model loaded successfully!")

    def _warmup(self):
        """
        Run a dummy forward pass so XLA compiles the graph before
        the first real user query. Without this, the first query
        takes 60-120s for compilation and Streamlit times out.
        """
        print("Warming up TPU (compiling XLA graph)...")
        dummy = self.tokenizer("Hello", return_tensors="pt", return_attention_mask=False)
        dummy_ids = dummy["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=dummy_ids, use_cache=False)
        xm.mark_step()
        print("TPU warmup complete.")

    def generate_response(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise research assistant. "
                    "Answer ONLY from the provided context. "
                    "If the context lacks the answer, say so in one sentence."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_ids = inputs["input_ids"].to(self.device)
        print(f"Input token count: {input_ids.shape[-1]}")

        with torch.no_grad():
            output_ids = self._greedy_generate_cached(input_ids)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def _greedy_generate_cached(self, input_ids):
        eos_id = self.tokenizer.eos_token_id
        generated = input_ids  # stays on TPU throughout
        next_token = None

        # Prefill pass — process the full prompt
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # on TPU
        xm.mark_step()

        generated = torch.cat([generated, next_token], dim=-1)  # both on TPU

        if next_token.item() == eos_id:
            return generated

        # Decode loop — single token at a time using KV cache
        for step in range(self.max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,  # TPU tensor
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )  # output is on TPU

            if step % 10 == 0:
                xm.mark_step()

            # BOTH tensors are on TPU — no .cpu() anywhere in the loop
            generated = torch.cat([generated, next_token], dim=-1)  #

            # Pull scalar to CPU only for the EOS comparison (cheap, just 1 int)
            if next_token.item() == eos_id:
                print(f"EOS hit at step {step + 1}")
                break

        xm.mark_step()  # final sync before returning
        return generated