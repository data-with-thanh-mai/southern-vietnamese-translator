import torch
from torch.utils.data import Dataset, DataLoader

class DialectDataset(Dataset):
    """
    Lớp xử lý dữ liệu tùy chỉnh cho bài toán dịch phương ngữ Miền Tây sang Phổ thông.
    """
    def __init__(self, dataframe, tokenizer, max_source_len=256, max_target_len=256):
        """
        Khởi tạo Dataset.
        Args:
            dataframe (pd.DataFrame): Dữ liệu dạng bảng chứa 2 cột đã làm sạch là 'source_text' và 'target_text'.
            tokenizer: HuggingFace Tokenizer (ví dụ: viT5 hoặc BARTPho tokenizer).
            max_source_len (int): Độ dài tối đa cho chuỗi đầu vào.
            max_target_len (int): Độ dài tối đa cho chuỗi đầu ra (nhãn).
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        source_text = str(item['source_text'])
        target_text = str(item['target_text'])

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # KỸ THUẬT CHE NHÃN (LABEL MASKING)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

def create_dataloader(dataframe, tokenizer, batch_size, max_source_len=256, max_target_len=256, is_train=True):
    """
    Hàm khởi tạo DataLoader để đưa dữ liệu vào mô hình theo từng batch.
    """
    dataset = DialectDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_source_len=max_source_len,
        max_target_len=max_target_len
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,       
        num_workers=2,          # Sử dụng đa luồng để nạp dữ liệu nhanh hơn
        pin_memory=True         # Tối ưu hóa tốc độ chuyển tensor từ CPU sang GPU
    )
