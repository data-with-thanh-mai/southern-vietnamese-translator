import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def build_lora_model(cfg, device="cuda", checkpoint_path=None):
    """
    Hàm khởi tạo hoặc nạp mô hình viT5 bọc LoRA.
    - Chạy lúc Train: Không truyền checkpoint_path -> Tạo LoRA mới.
    - Chạy lúc Test: Truyền checkpoint_path -> Load trọng số đã học.
    """
    print(f" Đang khởi tạo viT5 (Base Model: {cfg['model_name']})...")
    
    # 1. Tải tokenizer và base model 
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    
    # 2. KIỂM TRA CHẾ ĐỘ (TRAIN HAY TEST)
    if checkpoint_path is not None:
        # ==========================================
        # CHẾ ĐỘ TEST (EVALUATE): Nạp lại "sổ tay" đã ghi chép
        # ==========================================
        print(f" Đang nạp trọng số LoRA từ Checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
    else:
        # ==========================================
        # CHẾ ĐỘ TRAIN: Phát "sổ tay" trắng để bắt đầu học
        # ==========================================
        print(f" Đang gắn sổ tay LoRA mới (r={cfg['lora_r']}) để Train...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg["lora_r"],                                     
            lora_alpha=cfg["lora_alpha"],              
            target_modules=cfg["target_modules"], 
            lora_dropout=cfg["lora_dropout"],          
            bias=cfg["bias"]                            
        )
        model = get_peft_model(base_model, lora_config)
        
        # Chỉ in thông số khi Train
        print("\n" + "="*50)
        print(" KIỂM TRA THÔNG SỐ HUẤN LUYỆN LORA")
        print("="*50)
        model.print_trainable_parameters()
        print("="*50 + "\n")
    
    return model.to(device), tokenizer
