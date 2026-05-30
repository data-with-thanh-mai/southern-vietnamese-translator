import torch
from torch.utils.data import Dataset, DataLoader

class DialectDataset(Dataset):
    """
    Lớp xử lý dữ liệu tùy chỉnh cho bài toán dịch phương ngữ Miền Tây sang Phổ thông.
    Đã được nâng cấp để hỗ trợ cả HuggingFace Tokenizer (viT5) và Custom Tokenizer (LSTM).
    """
    def __init__(self, dataframe, tokenizer, max_source_len=256, max_target_len=256, model_type="transformer_full"):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.model_type = model_type # Biến cờ hiệu để phân luồng

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        source_text = str(item['input_text'])
        target_text = str(item['target_text'])

        # ==========================================
        # NHÁNH 1: XỬ LÝ CHO TRANSFORMER (viT5 / LoRA)
        # ==========================================
        if self.model_type in ["transformer_full", "transformer_lora"]:
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

            return {
                'input_ids': source_encoding['input_ids'].squeeze(),
                'attention_mask': source_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }

        # ==========================================
        # NHÁNH 2: XỬ LÝ CHO LSTM (Của Minh)
        # ==========================================
        elif self.model_type == "lstm":
            # Chuyển chuỗi thành danh sách các token ID
            src_ids = self.tokenizer.encode(source_text)
            trg_ids = self.tokenizer.encode(target_text)

            # Cắt ngắn nếu câu dài hơn mức cho phép (Truncation thủ công)
            src_ids = src_ids[:self.max_source_len]
            trg_ids = trg_ids[:self.max_target_len]

            # Lấy padding index (Mặc định là 0 nếu không tìm thấy)
            pad_idx = getattr(self.tokenizer, 'pad_token_id', 0)
            
            # Bơm thêm số 0 (PAD) vào cuối cho đủ độ dài (Padding thủ công)
            src_ids = src_ids + [pad_idx] * (self.max_source_len - len(src_ids))
            trg_ids = trg_ids + [pad_idx] * (self.max_target_len - len(trg_ids))

            return {
                'input_ids': torch.tensor(src_ids, dtype=torch.long),
                # Mask thủ công: 1 cho token thật, 0 cho token PAD
                'attention_mask': torch.tensor([1 if token != pad_idx else 0 for token in src_ids], dtype=torch.long),
                'labels': torch.tensor(trg_ids, dtype=torch.long)
            }

def create_dataloader(dataframe, tokenizer, batch_size, max_source_len=256, max_target_len=256, is_train=True, model_type="transformer_full"):
    """
    Hàm khởi tạo DataLoader để đưa dữ liệu vào mô hình theo từng batch.
    """
    dataset = DialectDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        model_type=model_type # Truyền cờ hiệu xuống Dataset
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,       
        num_workers=2,          
        pin_memory=True         
    )
