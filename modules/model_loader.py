"""
Module for loading the Qwen2.5-3B base model from Hugging Face.

Supports:
  - Loading model and tokenizer with default or custom configuration
  - Selecting dtype (float16 / bfloat16 / float32)
  - Selecting device (auto / cpu / cuda)
  - Loading model with 4-bit quantization (BitsAndBytes) to save VRAM
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

# ── Default constants ────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models"

# Force Hugging Face to use the local models/ directory instead of ~/.cache/huggingface
os.environ["HF_HOME"] = str(DEFAULT_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(DEFAULT_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(DEFAULT_CACHE_DIR)


def resolve_model_path(model_name: str) -> str:
    """Return the absolute path if model_name is a subdirectory inside models/.
    Otherwise, keep model_name as-is to load from Hugging Face Hub."""
    path_direct = Path(model_name)
    if path_direct.exists() and path_direct.is_dir():
        return str(path_direct.resolve())

    local_path = DEFAULT_CACHE_DIR / model_name
    name_only = model_name.split("/")[-1]
    local_path_name_only = DEFAULT_CACHE_DIR / name_only

    if local_path.exists() and local_path.is_dir():
        logger.info("Using local model (exact path): %s", local_path)
        return str(local_path.resolve())
    elif local_path_name_only.exists() and local_path_name_only.is_dir():
        logger.info("Using local model (short directory name): %s", local_path_name_only)
        return str(local_path_name_only.resolve())

    return model_name


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert a dtype string to a torch.dtype."""
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
            f"Invalid dtype '{dtype_str}'. "
            f"Supported values: {list(mapping.keys())}"
        )
    return mapping[dtype_str]


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: Optional[str] = None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """Create a 4-bit BitsAndBytes quantization config.

    Args:
        load_in_4bit: Enable 4-bit loading.
        bnb_4bit_compute_dtype: Compute dtype (auto-detected from hardware if None).
        bnb_4bit_quant_type: Quantization scheme (nf4 / fp4).
        bnb_4bit_use_double_quant: Use double quantization to save additional memory.

    Returns:
        A configured BitsAndBytesConfig instance.
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
    """Load the tokenizer for Qwen2.5-3B.

    Args:
        model_name: Model name on Hugging Face Hub.
        cache_dir: Cache directory (None = HF default).
        trust_remote_code: Allow executing code from the model repository.
        padding_side: Padding side ("right" or "left").

    Returns:
        A configured tokenizer instance.
    """
    resolved_model_name = resolve_model_path(model_name)
    logger.info("Loading tokenizer: %s", resolved_model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    tokenizer.padding_side = padding_side

    # Ensure pad_token exists (some models do not include one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token was missing; assigned eos_token as pad_token.")

    logger.info("Tokenizer loaded successfully. Vocab size: %d", tokenizer.vocab_size)
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
    """Load the Qwen2.5-3B base model.

    Args:
        model_name: Model name on Hugging Face Hub.
        cache_dir: Cache directory (None = HF default).
        trust_remote_code: Allow executing code from the model repository.
        torch_dtype: Data type (float16 / bfloat16 / float32).
        device_map: Device placement strategy ("auto" / "cpu" / "cuda").
        quantization_config: BitsAndBytes config (None = no quantization).
        attn_implementation: Attention backend ("flash_attention_2", "sdpa", …).

    Returns:
        The model loaded onto the target device.
    """
    resolved_model_name = resolve_model_path(model_name)
    logger.info("Loading model: %s", resolved_model_name)
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

    # Log basic model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded successfully. Total params: %s | Trainable: %s",
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
    """Load both the model and tokenizer in a single call.

    Args:
        model_name: Model name on Hugging Face Hub.
        cache_dir: Cache directory.
        trust_remote_code: Allow executing code from the model repository.
        torch_dtype: Data type.
        device_map: Device placement strategy.
        use_4bit: Enable 4-bit quantization (saves ~60% VRAM).
        attn_implementation: Attention backend.
        padding_side: Tokenizer padding side.

    Returns:
        A tuple of (model, tokenizer).
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


# ── Quick CLI test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    model, tokenizer = load_model_and_tokenizer(use_4bit=True)

    # Generate a short text sample to verify the model works
    prompt = "Meeting summary:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
