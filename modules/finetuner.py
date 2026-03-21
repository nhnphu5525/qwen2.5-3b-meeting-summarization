import os
import torch
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

def prepare_model_for_lora(model: PreTrainedModel, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """Chuẩn bị mô hình để huấn luyện bằng QLoRA."""
    logger.info("Đang cấu hình mô hình cho QLoRA (4-bit training).")
    
    # Kích hoạt tính năng gradient checkpointing để tiết kiệm VRAM
    model.gradient_checkpointing_enable()
    
    # Chuẩn bị model cho việc train ở dạng k-bit (4-bit)
    model = prepare_model_for_kbit_training(model)
    
    # Cấu hình LoRA (chỉ fine-tune các linear layer quan trọng)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    
    # In ra số lượng tham số được train để kiểm tra
    model.print_trainable_parameters()
    return model

def get_training_arguments(output_dir: str = "models/finetuned-meeting-summary", max_seq_length: int = 1024, num_train_epochs: int = 3):
    """Tạo đối tượng SFTConfig với config tối ưu cho GPU cá nhân lấy tương thích với thư viện trl mới nhất."""
    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=1,        # Nhỏ nhất có thể để tránh OOM
        gradient_accumulation_steps=4,        # Cộng dồn gradient để tăng batch size lý thuyết lên 4
        learning_rate=2e-4,
        logging_steps=5,                      # Số bước in log
        num_train_epochs=num_train_epochs,    # Quét qua toàn bộ dữ liệu theo số vòng lấy từ tham số
        save_steps=25,                        # Lưu backup sau mỗi 25 step
        fp16=not is_bf16,                     # Tự động dùng fp16 trên GPU đời cũ
        bf16=is_bf16,                         # Tự động dùng bf16 trên GPU đời mới để không lỗi kiểu dữ liệu
        optim="paged_adamw_8bit",             # Dùng optimizer 8-bit để tránh OOM
        report_to="none",                     # Không báo cáo lên wandb
    )
    
    # Try to set max_seq_length safely
    if hasattr(config, 'max_seq_length'):
        config.max_seq_length = max_seq_length
    else:
        # Tương thích ngược với phiên bản TRL cũ hơn
        setattr(config, 'max_seq_length', max_seq_length)
        
    # Cài đặt tham số đánh giá (Evaluation) tự động nếu có
    config.eval_strategy = "steps"
    config.eval_steps = 25 # Đánh giá sau mỗi 25 step
        
    return config

def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    training_args: SFTConfig,
    eval_dataset=None,
    max_seq_length: int = 1024
):
    """Tạo đối tượng SFTTrainer từ thư viện trl."""
    logger.info("Đang khởi tạo SFTTrainer.")
    
    # Cập nhật max_seq_length nếu có cung cấp từ config notebook
    if hasattr(training_args, 'max_seq_length'):
        training_args.max_seq_length = max_seq_length
    else:
        setattr(training_args, 'max_seq_length', max_seq_length)

    # Nếu không có tập val, tắt chiến lược đánh giá
    if eval_dataset is None:
        training_args.eval_strategy = "no"

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    return trainer
