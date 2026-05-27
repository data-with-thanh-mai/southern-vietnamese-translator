import torch
from transformers import AutoModelForSeq2SeqLM

def build_transformer_full_ft(model_name="VietAI/vit5-base", device="cpu"):
    """
    Khởi tạo mô hình Transformer pre-trained cho quá trình Full Fine-Tuning.
    
    Args:
        model_name (str): Tên mô hình trên Hugging Face Hub (vd: 'VietAI/vit5-base' hoặc 'vinai/bartpho-word').
        device (str): Thiết bị tính toán ('cuda' hoặc 'cpu').
        
    Returns:
        model: Mô hình đã được mở khóa toàn bộ trọng số.
    """
    print(f"[*] Đang tải pre-trained model: {model_name}...")
    
    # Nạp mô hình Seq2Seq từ Hugging Face
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Đảm bảo toàn bộ trọng số được MỞ KHÓA (requires_grad = True) cho Full Fine-Tuning
    for param in model.parameters():
        param.requires_grad = True
        
    model.to(device)
    print("[*] Tải mô hình thành công và đã đẩy lên thiết bị:", device)
    
    return model

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


