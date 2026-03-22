import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


class TPULLMEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading {model_id} onto TPU...")

        self.device = xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        print("TPU Model loaded successfully!")

    def generate_response(self, prompt, max_new_tokens=512):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI research assistant. Answer based on the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize — NO attention mask (avoids float-mask bitwise_or crash on XLA)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids = inputs["input_ids"].to(self.device)

        # Greedy decoding only — do_sample=True triggers sampling paths that
        # perform bitwise_or on float32 masks, which is illegal in XLA and
        # causes a SIGABRT. Greedy is stable and fast on TPU.
        with torch.no_grad():
            output_ids = self._greedy_generate(input_ids, max_new_tokens)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def _greedy_generate(self, input_ids, max_new_tokens):
        """
        Manual greedy token-by-token generation loop.
        Avoids transformers' internal generate() code paths that create
        float attention masks and call bitwise_or — fatal on torch_xla.
        """
        eos_id = self.tokenizer.eos_token_id
        generated = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated, use_cache=False)

            # Greedy: pick the highest-logit token at the last position
            next_token_logits = outputs.logits[:, -1, :]          # (1, vocab)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)

            # Sync TPU step before appending — keeps XLA graph small
            xm.mark_step()

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS
            if next_token.item() == eos_id:
                break

        return generated
