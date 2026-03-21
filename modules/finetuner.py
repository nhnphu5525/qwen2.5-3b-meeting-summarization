import os
import torch
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

def prepare_model_for_lora(model: PreTrainedModel, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """Prepare the model for QLoRA fine-tuning."""
    logger.info("Configuring model for QLoRA (4-bit training).")
    
    # Enable gradient checkpointing to reduce VRAM usage
    model.gradient_checkpointing_enable()
    
    # Prepare the model for k-bit (4-bit) training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA (only fine-tune key linear layers)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    
    # Print the number of trainable parameters for verification
    model.print_trainable_parameters()
    return model

def get_training_arguments(output_dir: str = "models/finetuned-meeting-summary", max_seq_length: int = 1024, num_train_epochs: int = 3):
    """Create an SFTConfig object with settings optimized for a personal GPU, compatible with the latest trl library."""
    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=1,        # Minimum batch size to avoid OOM
        gradient_accumulation_steps=4,        # Accumulate gradients to effectively increase batch size to 4
        learning_rate=2e-4,
        logging_steps=5,                      # Number of steps between log outputs
        num_train_epochs=num_train_epochs,    # Number of full passes over the dataset
        save_steps=25,                        # Save a checkpoint every 25 steps
        fp16=not is_bf16,                     # Use fp16 automatically on older GPUs
        bf16=is_bf16,                         # Use bf16 automatically on newer GPUs to avoid dtype errors
        optim="paged_adamw_8bit",             # Use 8-bit optimizer to prevent OOM
        report_to="none",                     # Disable reporting to wandb
    )
    
    # Try to set max_seq_length safely
    if hasattr(config, 'max_seq_length'):
        config.max_seq_length = max_seq_length
    else:
        # Backward compatibility with older TRL versions
        setattr(config, 'max_seq_length', max_seq_length)
        
    # Configure automatic evaluation settings if supported
    config.eval_strategy = "steps"
    config.eval_steps = 25  # Evaluate every 25 steps
        
    return config

def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    training_args: SFTConfig,
    eval_dataset=None,
    max_seq_length: int = 1024
):
    """Create an SFTTrainer instance from the trl library."""
    logger.info("Initializing SFTTrainer.")
    
    # Apply max_seq_length from the notebook config if provided
    if hasattr(training_args, 'max_seq_length'):
        training_args.max_seq_length = max_seq_length
    else:
        setattr(training_args, 'max_seq_length', max_seq_length)

    # Disable evaluation strategy if no validation set is provided
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
