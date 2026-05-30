import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def build_lora_model(cfg, device="cuda"):
    """
    Hàm khởi tạo mô hình viT5 bọc LoRA dựa trên cấu hình từ config.py
    """
    print(f"🤖 Đang khởi tạo viT5 với LoRA (r={cfg['lora_r']})...")
    
    # 1. Tải tokenizer và base model
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    
    # 2. Cấu hình LoRA lấy từ config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["lora_r"],                             
        lora_alpha=cfg["lora_alpha"],             
        target_modules=cfg["target_modules"], 
        lora_dropout=cfg["lora_dropout"],          
        bias=cfg["bias"]                            
    )
    
    # 3. Tích hợp LoRA
    model = get_peft_model(base_model, lora_config)
    
    # 4. In thông số
    print("\n" + "="*50)
    print(" KIỂM TRA THÔNG SỐ HUẤN LUYỆN LORA")
    print("="*50)
    model.print_trainable_parameters()
    print("="*50 + "\n")
    
    return model.to(device), tokenizer
