import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from source.data.data_loader import create_dataloader
from source.models.seq2seq_lstm import Seq2Seq
import config

def build_optimizer_and_scheduler(model: nn.Module, cfg: dict, total_steps: int) -> tuple:
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": getattr(cfg, "WEIGHT_DECAY", 1e-4),
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=cfg.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    warmup_steps = int(total_steps * getattr(cfg, "WARMUP_RATIO", 0.1))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def train_model(model, train_loader, val_loader, tokenizer, config, device="cuda"):
    print("\n🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN.")
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0 
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    if config.MODEL_TYPE == "lstm":
        criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    for epoch in range(config.NUM_EPOCHS):
        # ==========================================
        # 1. PHA HUẤN LUYỆN (TRAINING)
        # ==========================================
        model.train() 
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for step, batch in enumerate(train_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            labels = batch['labels'].to(device)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            
            if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
                # Chuẩn bị label cho HuggingFace: bỏ qua PAD token
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif config.MODEL_TYPE == "lstm":
                # Kích hoạt teacher forcing và truyền pad_idx
                outputs = model(src=input_ids, trg=labels, pad_idx=config.PAD_IDX, tf_ratio=config.TF_RATIO)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], global_step)
            
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        
        # ==========================================
        # 2. PHA ĐÁNH GIÁ (VALIDATION)
        # ==========================================
        model.eval() 
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad(): 
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels'].to(device)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
    
                if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
                    labels[labels == tokenizer.pad_token_id] = -100
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                elif config.MODEL_TYPE == "lstm":
                    # TẮT teacher forcing (tf_ratio = 0.0) khi validate
                    outputs = model(src=input_ids, trg=labels, pad_idx=config.PAD_IDX, tf_ratio=0.0)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch)
        
        # ==========================================
        # 3. LOG & EARLY STOPPING 
        # ==========================================
        print(f"Kết quả Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"🔥 KỶ LỤC MỚI! Val Loss giảm từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            
            checkpoint_dir = os.path.join("outputs", "checkpoints", "best_model")
            os.makedirs(checkpoint_dir, exist_ok=True) 
            
            if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir) 
            elif config.MODEL_TYPE == "lstm":
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "lstm_best.pth"))
                
            print(f"💾 Đã lưu thành công mô hình tốt nhất vào: {checkpoint_dir}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Val Loss không giảm. Cảnh báo Overfitting lần {epochs_no_improve}/{config.PATIENCE}.")
            
            if epochs_no_improve >= config.PATIENCE:
                print(f"🛑 ĐÃ KÍCH HOẠT EARLY STOPPING! Dừng huấn luyện sớm ở Epoch {epoch+1}.")
                break 
            
    writer.close()
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN ĐÃ KẾT THÚC!")
    return model


# ==============================================================================
# KHỐI KHỞI CHẠY THỰC TẾ (EntryPoint)
# ==============================================================================
if __name__ == "__main__":
    import pandas as pd
    import random
    import numpy as np
    
    # 0. CỐ ĐỊNH SEED 
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Đang sử dụng thiết bị: {device.upper()}")

    # 1. IMPORT DATASET 
    print("⏳ Đang import dữ liệu từ ổ đĩa...")
    try:
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH) # Sửa lỗi Data Leakage ở đây
        print(f"✅ Đã nạp thành công: {len(train_df)} câu Train | {len(val_df)} câu Val")
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy dữ liệu! Vui lòng kiểm tra lại đường dẫn trong config.py\nChi tiết: {e}")
        exit()
    
    # 2. KHỞI TẠO TOKENIZER & MODEL 
    print(f"🤖 Đang chuẩn bị mô hình: {config.MODEL_TYPE.upper()}...")
    
    if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        tokenizer = AutoTokenizer.from_pretrained(config.active_cfg["model_name"],use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.active_cfg["model_name"])
        
        if config.MODEL_TYPE == "transformer_lora":
            from peft import get_peft_model, LoraConfig, TaskType
            print("🗜️ Đang bọc lớp LoRA siêu nhẹ vào mô hình...")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=config.active_cfg["lora_r"],
                lora_alpha=config.active_cfg["lora_alpha"],
                lora_dropout=config.active_cfg["lora_dropout"],
                target_modules=config.active_cfg["target_modules"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters() 
            
    elif config.MODEL_TYPE == "lstm":
        # Lưu ý: Cần import WordLevelTokenizer (hoặc custom tokenizer của Minh) để chạy DataLoader
        # Nếu Minh đặt tên class khác thì bạn nhớ đổi tên lại nha
        from source.data.data_loader import WordLevelTokenizer
        from source.models.seq2seq_lstm import Encoder, Decoder
        
        tokenizer = WordLevelTokenizer.load_from_json(config.VOCAB_PATH)
        cfg = config.active_cfg
        
        encoder = Encoder(
            vocab_size = cfg["vocab_size"],
            embed_dim  = cfg["embed_dim"],
            hidden_dim = cfg["hidden_dim"],
            n_layers   = cfg["n_layers"],
            dropout    = cfg["dropout"],
            pad_idx    = cfg["pad_idx"]
        )
        decoder = Decoder(
            vocab_size  = cfg["vocab_size"],
            embed_dim   = cfg["embed_dim"],
            hidden_dim  = cfg["hidden_dim"],
            encoder_dim = cfg["hidden_dim"] * 2,  
            n_layers    = cfg["n_layers"],
            dropout     = cfg["dropout"],
            pad_idx     = cfg["pad_idx"]
        )
        model = Seq2Seq(encoder, decoder, device)
        print(f"✅ Khởi tạo thành công mạng LSTM ({cfg['n_layers']} layers).")
        
    model = model.to(device)

    # 3. ĐÓNG GÓI DATALOADER
    print("📦 Đang đóng gói dữ liệu vào DataLoader...")
    train_loader = create_dataloader(
        dataframe=train_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=True,
        model_type=config.MODEL_TYPE # TRUYỀN CỜ HIỆU VÀO ĐÂY
    )
    
    val_loader = create_dataloader(
        dataframe=val_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=False,
        model_type=config.MODEL_TYPE # TRUYỀN CỜ HIỆU VÀO ĐÂY
    )

    # 4. KÍCH HOẠT HUẤN LUYỆN
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
