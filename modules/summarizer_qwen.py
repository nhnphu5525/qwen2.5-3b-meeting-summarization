import threading
from modules.model_loader import load_model_and_tokenizer
import torch

# Template trống dùng cho lần tóm tắt đầu tiên (giống training data v2 part1)
_EMPTY_TEMPLATE = """# **Tiêu đề**
## I. Nội dung chính

### 1. Mục tiêu cuộc họp

### 2. Các vấn đề đã thảo luận

### 3. Kết luận và quyết định

## II. Danh sách công việc cần làm

| Công việc | Người phụ trách | Hạn chót |
| :--- | :--- | :--- |
| | ||
| | ||
| | ||
| | ||
| | ||"""


class QwenSummarizer:
    def __init__(self, model_dir="models/qwen25-3b-v2", device=None,
                 max_input_tokens=2048, max_output_tokens=1024):
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
        # State cho tóm tắt tăng dần
        self._previous_summary: str = ""
        self._last_summarized_tokens: int = 0

    def _build_prompt(self, input_text: str) -> str:
        """Build prompt khớp với format training data v2."""
        return (
            "Hãy cập nhật lại báo cáo cuộc họp trước đó bằng cách bổ sung thêm "
            "thông tin từ nội dung thảo luận mới dưới đây.\n\n"
            f"### Đầu vào (Báo cáo cũ & Nội dung thảo luận mới):\n"
            f"{input_text.strip()}\n\n"
            "### Báo cáo cuộc họp đã cập nhật:\n"
        )

    def summarize(self, transcript: str) -> str:
        """Tóm tắt tăng dần: lần đầu dùng template trống, lần sau dùng báo cáo trước đó."""
        # Xây dựng input giống training data
        if self._previous_summary:
            input_text = self._previous_summary + "\n" + transcript
        else:
            input_text = _EMPTY_TEMPLATE + "\n" + transcript

        prompt = self._build_prompt(input_text)

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_input_tokens,
        ).to(self.model.device)

        with torch.no_grad(), self.lock:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode chỉ phần output mới (bỏ prompt)
        input_len = inputs["input_ids"].shape[1]
        summary = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Lưu lại summary cho lần tóm tắt tăng dần tiếp theo
        if summary:
            self._previous_summary = summary
            self._last_summarized_tokens = input_len  # Lưu số token đã xử lý

        return summary

    def reset(self):
        """Reset state khi người dùng bấm Clear."""
        self._previous_summary = ""
        self._last_summarized_tokens = 0
