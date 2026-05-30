import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def build_transformer_full_ft(model_name="VietAI/vit5-base", device="cpu", use_fast=False):
    """
    Khởi tạo mô hình Transformer pre-trained và Tokenizer cho quá trình Full Fine-Tuning.
    
    Args:
        model_name (str): Tên mô hình trên Hugging Face Hub.
        device (str): Thiết bị tính toán ('cuda' hoặc 'cpu').
        use_fast (bool): Tắt Fast Tokenizer để tránh lỗi KeyError: 0 của viT5.
        
    Returns:
        model, tokenizer: Mô hình đã mở khóa và bộ băm từ chuẩn.
    """
    print(f"[*] Đang tải pre-trained model và tokenizer: {model_name}...")
    
    # 1. Tải Tokenizer và TẮT chế độ fast
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    
    # 2. Nạp mô hình Seq2Seq từ Hugging Face
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. Đảm bảo toàn bộ trọng số được MỞ KHÓA (requires_grad = True) cho Full Fine-Tuning
    for param in model.parameters():
        param.requires_grad = True
        
    model.to(device)
    print("[*] Tải mô hình thành công và đã đẩy lên thiết bị:", device)
    
    return model, tokenizer

def count_parameters(model):
    """
    Tính toán và in ra tổng số tham số của mạng.
    """
    # Lấy tổng số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    
    # Lấy số lượng tham số được phép cập nhật gradient (Trainable)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print(f"Tổng số tham số của mô hình: {total_params:,}")
    print(f"Số tham số huấn luyện (Trainable): {trainable_params:,}")
    print(f"Tỷ lệ tham số huấn luyện: {(trainable_params / total_params) * 100:.2f}%")
    print("-" * 50)
    
    return total_params, trainable_params
