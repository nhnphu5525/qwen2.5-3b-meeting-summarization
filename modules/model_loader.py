"""
Module tải base model Qwen2.5-3B từ Hugging Face.

Hỗ trợ:
  - Tải model và tokenizer với cấu hình mặc định hoặc tùy chỉnh
  - Chọn dtype (float16 / bfloat16 / float32)
  - Chọn device (auto / cpu / cuda)
  - Tải model theo chế độ 4-bit quantization (BitsAndBytes) để tiết kiệm VRAM
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# ── Hằng số mặc định ────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models"

# Ép buộc Hugging Face dùng thư mục models/ thay vì ~/.cache/huggingface mặc định
os.environ["HF_HOME"] = str(DEFAULT_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(DEFAULT_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(DEFAULT_CACHE_DIR)


def resolve_model_path(model_name: str) -> str:
    """Trả về đường dẫn tuyệt đối nếu model_name là thư mục con trong models/.
    Nếu không có, giữ nguyên model_name để tải từ Hugging Face."""
    path_direct = Path(model_name)
    if path_direct.exists() and path_direct.is_dir():
        return str(path_direct.resolve())

    local_path = DEFAULT_CACHE_DIR / model_name
    name_only = model_name.split("/")[-1]
    local_path_name_only = DEFAULT_CACHE_DIR / name_only

    if local_path.exists() and local_path.is_dir():
        logger.info("Sử dụng model local (chỉ định chính xác): %s", local_path)
        return str(local_path.resolve())
    elif local_path_name_only.exists() and local_path_name_only.is_dir():
        logger.info("Sử dụng model local (theo tên thư mục rút gọn): %s", local_path_name_only)
        return str(local_path_name_only.resolve())

    return model_name


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Chuyển chuỗi dtype thành torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype_str = dtype_str.lower().strip()
    if dtype_str not in mapping:
        raise ValueError(
            f"dtype '{dtype_str}' không hợp lệ. "
            f"Các giá trị hỗ trợ: {list(mapping.keys())}"
        )
    return mapping[dtype_str]


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: Optional[str] = None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """Tạo cấu hình quantization 4-bit (BitsAndBytes).

    Args:
        load_in_4bit: Bật chế độ 4-bit.
        bnb_4bit_compute_dtype: Kiểu dữ liệu tính toán (tự động theo phần cứng).
        bnb_4bit_quant_type: Phương pháp quantize (nf4 / fp4).
        bnb_4bit_use_double_quant: Sử dụng double quantization để tiết kiệm thêm bộ nhớ.

    Returns:
        BitsAndBytesConfig đã được cấu hình.
    """
    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
        
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=_resolve_dtype(bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = str(DEFAULT_CACHE_DIR),
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Tải tokenizer của Qwen2.5-3B.

    Args:
        model_name: Tên model trên Hugging Face Hub.
        cache_dir: Thư mục cache (None = mặc định HF).
        trust_remote_code: Cho phép chạy code từ model repo.
        padding_side: Hướng padding ("right" hoặc "left").

    Returns:
        Tokenizer đã được cấu hình.
    """
    resolved_model_name = resolve_model_path(model_name)
    logger.info("Đang tải tokenizer: %s", resolved_model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    tokenizer.padding_side = padding_side

    # Đảm bảo có pad_token (một số model không có sẵn)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token chưa có, đã gán bằng eos_token.")

    logger.info("Tải tokenizer thành công. Vocab size: %d", tokenizer.vocab_size)
    return tokenizer


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = str(DEFAULT_CACHE_DIR),
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    quantization_config: Optional[BitsAndBytesConfig] = None,
    attn_implementation: Optional[str] = None,
) -> PreTrainedModel:
    """Tải base model Qwen2.5-3B.

    Args:
        model_name: Tên model trên Hugging Face Hub.
        cache_dir: Thư mục cache (None = mặc định HF).
        trust_remote_code: Cho phép chạy code từ model repo.
        torch_dtype: Kiểu dữ liệu (float16 / bfloat16 / float32).
        device_map: Cách phân bổ device ("auto" / "cpu" / "cuda").
        quantization_config: Cấu hình BitsAndBytes (None = không quantize).
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", …).

    Returns:
        Model đã được tải lên device.
    """
    resolved_model_name = resolve_model_path(model_name)
    logger.info("Đang tải model: %s", resolved_model_name)
    logger.info(
        "  dtype=%s | device_map=%s | quantized=%s",
        torch_dtype,
        device_map,
        quantization_config is not None,
    )

    kwargs: dict = {
        "pretrained_model_name_or_path": resolved_model_name,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _resolve_dtype(torch_dtype),
        "device_map": device_map,
    }

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    # Log thông tin cơ bản
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Tải model thành công. Tổng params: %s | Trainable: %s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    return model


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = str(DEFAULT_CACHE_DIR),
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    use_4bit: bool = False,
    attn_implementation: Optional[str] = None,
    padding_side: str = "right",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Tải cả model và tokenizer trong một lời gọi duy nhất.

    Args:
        model_name: Tên model trên Hugging Face Hub.
        cache_dir: Thư mục cache.
        trust_remote_code: Cho phép chạy code từ model repo.
        torch_dtype: Kiểu dữ liệu.
        device_map: Cách phân bổ device.
        use_4bit: Bật quantization 4-bit (tiết kiệm ~60 % VRAM).
        attn_implementation: Attention implementation.
        padding_side: Hướng padding cho tokenizer.

    Returns:
        Tuple (model, tokenizer).
    """
    quantization_config = get_quantization_config() if use_4bit else None

    tokenizer = load_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )

    model = load_model(
        model_name=model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )

    return model, tokenizer


# ── CLI nhanh để test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    model, tokenizer = load_model_and_tokenizer(use_4bit=True)

    # Sinh thử một đoạn text ngắn
    prompt = "Meeting summary:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
