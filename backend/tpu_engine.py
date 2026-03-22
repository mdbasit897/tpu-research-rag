import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class TPULLMEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-32B-Instruct"):
        print(f"Loading {model_id} onto TPU...")

        # Initialize the TPU device
        self.device = xm.xla_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model in bfloat16 for optimal TPU performance
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device  # Maps weights directly to the TPU
        )

        # Create a text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.3,
            device=self.device
        )
        print("TPU Model loaded successfully!")

    def generate_response(self, prompt):
        # Format for Qwen's instruct template
        messages = [
            {"role": "system",
             "content": "You are a helpful AI research assistant. Answer based on the provided context."},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = self.pipe(prompt_text)
        # Strip the input prompt from the final output
        response = outputs[0]["generated_text"].split("<|im_start|>assistant\n")[-1]
        return response.strip()