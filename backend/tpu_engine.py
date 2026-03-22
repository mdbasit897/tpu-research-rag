import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


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

        self.max_new_tokens = 256   # Keep low to avoid HBM OOM
        self.max_input_tokens = 1024  # Hard cap on input length

        print("TPU Model loaded successfully!")

    def generate_response(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI research assistant. Be concise. Answer based on the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize with hard truncation to avoid OOM from long RAG contexts
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
        """
        Greedy generation with dynamic KV cache.
        - use_cache=True: only processes NEW tokens each step (O(n) not O(n²))
        - xm.mark_step() after each token: keeps XLA graphs small, avoids recompilation
        - Stops at EOS or max_new_tokens
        """
        eos_id = self.tokenizer.eos_token_id
        past_key_values = None
        generated = input_ids
        current_input = input_ids  # First step: full prompt. After: only new token.

        for step in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,          # Reuse KV cache — critical for memory
                )

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Sync TPU graph every step to prevent graph explosion
            xm.mark_step()

            generated = torch.cat([generated, next_token], dim=-1)

            # Next iteration only feeds the single new token
            current_input = next_token

            if next_token.item() == eos_id:
                print(f"EOS hit at step {step + 1}")
                break

        return generated