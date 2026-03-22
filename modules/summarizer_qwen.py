import threading
from modules.model_loader import load_model_and_tokenizer
import torch

class QwenSummarizer:
    def __init__(self, model_dir="models/qwen25-3b-v2", device=None, max_input_tokens=512, max_output_tokens=128):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_dir,
            device_map=device,
            torch_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
            use_4bit=True,
            padding_side="left"
        )
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.lock = threading.Lock()

    def summarize(self, transcript: str) -> str:
        prompt = f"Tóm tắt nội dung cuộc họp sau bằng tiếng Việt:\n{transcript}\nTóm tắt:"  # You can adjust prompt for your finetune
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.model.device)
        with torch.no_grad(), self.lock:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split("Tóm tắt:")[-1].strip() if "Tóm tắt:" in summary else summary.strip()
