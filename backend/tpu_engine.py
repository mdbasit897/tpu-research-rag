import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


class TPULLMEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading {model_id} onto TPU...")

        # Initialize the TPU device
        self.device = xm.xla_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model in bfloat16 for optimal TPU performance
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        print("TPU Model loaded successfully!")

    def generate_response(self, prompt):
        # Format for Qwen's instruct template
        messages = [
            {"role": "system",
             "content": "You are a helpful AI research assistant. Answer based on the provided context."},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "attention_mask" in inputs:
            # TPU/XLA bitwise ops require integral/pred masks.
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        response = generated_text.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0]
        return response.strip()
