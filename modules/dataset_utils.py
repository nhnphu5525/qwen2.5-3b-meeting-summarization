import os
import glob
from datasets import Dataset

def parse_meeting_file(filepath: str):
    """Read a txt file and extract the input (transcript) and output (summary) sections."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = content.split('<|output|>')
    if len(parts) != 2:
        return None, None
    
    input_part = parts[0].replace('<|input|>', '').strip()
    output_part = parts[1].strip()
    
    return input_part, output_part

def format_prompt(input_text: str, output_text: str) -> str:
    """Build a standardized prompt string for fine-tuning, supporting both v1 and v2 data."""
    # Detect if the input contains a previous report (v2 has markdown headers like '# ' or '## ')
    is_v2 = "# " in input_text or "## I." in input_text
    
    if is_v2:
        prompt = f"""Hãy cập nhật lại báo cáo cuộc họp trước đó bằng cách bổ sung thêm thông tin từ nội dung thảo luận mới dưới đây.

### Đầu vào (Báo cáo cũ & Nội dung thảo luận mới):
{input_text.strip()}

### Báo cáo cuộc họp đã cập nhật:
{output_text.strip()}"""
    else:
        prompt = f"""Hãy tóm tắt lại nội dung cuộc họp sau đây.

### Nội dung cuộc họp:
{input_text.strip()}

### Tóm tắt cuộc họp:
{output_text.strip()}"""

    return prompt

def load_and_prepare_dataset(data_dir: str):
    """Read all txt files in the given directory and return a Hugging Face Dataset."""
    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    texts = []
    for file in txt_files:
        input_text, output_text = parse_meeting_file(file)
        if input_text and output_text:
            # SFTTrainer expects a single 'text' field containing the full training content
            full_text = format_prompt(input_text, output_text)
            texts.append({"text": full_text})
            
    if not texts:
        raise ValueError(f"No valid data found in directory {data_dir}. Please check data/raw/")
        
    return Dataset.from_list(texts)

def load_and_split_dataset(data_dir: str, val_size=0.1, test_size=0.1, random_seed=42):
    """Load and split the dataset into three subsets: train, val, and test."""
    from datasets import DatasetDict
    full_dataset = load_and_prepare_dataset(data_dir)
    
    # Step 1: Split off the test set from the full dataset
    train_val_test = full_dataset.train_test_split(test_size=test_size, seed=random_seed)
    
    # Step 2: Split the validation set from the remaining (train + val) portion.
    # val_size is expressed as a fraction of the full dataset, so we rescale for the remaining split:
    val_ratio = val_size / (1.0 - test_size)
    
    train_val = train_val_test['train'].train_test_split(test_size=val_ratio, seed=random_seed)
    
    return DatasetDict({
        'train': train_val['train'],
        'val': train_val['test'],
        'test': train_val_test['test']
    })
