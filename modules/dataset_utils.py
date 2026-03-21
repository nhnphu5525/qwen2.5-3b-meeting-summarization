import os
import glob
from datasets import Dataset

def parse_meeting_file(filepath: str):
    """Đọc file txt và tách phần input (transcript) và output (summary)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = content.split('<|output|>')
    if len(parts) != 2:
        return None, None
    
    input_part = parts[0].replace('<|input|>', '').strip()
    output_part = parts[1].strip()
    
    return input_part, output_part

def format_prompt(input_text: str, output_text: str) -> str:
    """Tạo chuỗi prompt chuẩn cho quy trình fine-tune, hỗ trợ cả v1 và v2."""
    # Nhận diện nếu input chứa báo cáo cũ (v2 có chứa markdown header như '# ' hoặc '##')
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
    """Đọc tất cả các file txt trong thư mục và trả về Hugging Face Dataset."""
    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    texts = []
    for file in txt_files:
        input_text, output_text = parse_meeting_file(file)
        if input_text and output_text:
            # SFTTrainer yêu cầu một trường text chứa toàn bộ nội dung huấn luyện
            full_text = format_prompt(input_text, output_text)
            texts.append({"text": full_text})
            
    if not texts:
        raise ValueError(f"Không tìm thấy dữ liệu hợp lệ trong thư mục {data_dir}. Vui lòng kiểm tra lại data/raw/")
        
    return Dataset.from_list(texts)

def load_and_split_dataset(data_dir: str, val_size=0.1, test_size=0.1, random_seed=42):
    """Tải và chia dữ liệu thành 3 tập: train, val, test."""
    from datasets import DatasetDict
    full_dataset = load_and_prepare_dataset(data_dir)
    
    # Bước 1: Trích xuất tập test và (train + val)
    train_val_test = full_dataset.train_test_split(test_size=test_size, seed=random_seed)
    
    # Bước 2: Tách lấy val từ (train + val)
    # val_size được định nghĩa theo % của data gốc, nên tỉ lệ tính cho phần còn lại sẽ là:
    val_ratio = val_size / (1.0 - test_size)
    
    train_val = train_val_test['train'].train_test_split(test_size=val_ratio, seed=random_seed)
    
    return DatasetDict({
        'train': train_val['train'],
        'val': train_val['test'],
        'test': train_val_test['test']
    })
