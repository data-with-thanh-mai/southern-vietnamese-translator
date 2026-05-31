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
    
    # SETUP THƯ MỤC GHI LOG CHO TỪNG MODEL
    model_log_dir = os.path.join(config.LOG_DIR, config.MODEL_TYPE)
    os.makedirs(model_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # TẠO FILE TXT VÀ GHI TIÊU ĐỀ 
    log_file_path = os.path.join(model_log_dir, "training_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"=== NHẬT KÝ HUẤN LUYỆN: {config.MODEL_TYPE.upper()} ===\n")
        f.write("-" * 50 + "\n")
    
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
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif config.MODEL_TYPE == "lstm":
                outputs = model(src=input_ids, trg=labels, pad_idx=config.PAD_IDX, tf_ratio=config.TF_RATIO)
                loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))
            
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
                    outputs = model(src=input_ids, trg=labels, pad_idx=config.PAD_IDX, tf_ratio=0.0)
                    loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))
                
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch)
        
        # ==========================================
        # 3. LOG & EARLY STOPPING 
        # ==========================================
        print(f"Kết quả Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1:02d} | Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}\n")

        if avg_val_loss < best_val_loss:
            msg_improve = f"Val loss cải thiện từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}."
            print(msg_improve)
            
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"  -> {msg_improve} (Đã lưu checkpoint)\n")
                
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            
            folder_name = f"best_model_{config.MODEL_TYPE}"
            checkpoint_dir = os.path.join("outputs", "checkpoints", folder_name)
            os.makedirs(checkpoint_dir, exist_ok=True) 
            
            if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir) 
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{config.MODEL_TYPE}_best.pth"))
            elif config.MODEL_TYPE == "lstm":
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{config.MODEL_TYPE}_best.pth"))
                
            print(f"💾 Đã lưu mô hình tốt nhất vào: {checkpoint_dir}")
            
        else:
            epochs_no_improve += 1
            msg_warn = f"⚠️ Val Loss không giảm ({epochs_no_improve}/{config.PATIENCE})."
            print(msg_warn)
            
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"  -> {msg_warn}\n")
            
            if epochs_no_improve >= config.PATIENCE:
                msg_stop = f"🛑 KÍCH HOẠT EARLY STOPPING ở Epoch {epoch+1}."
                print(msg_stop)
                
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"  -> {msg_stop}\n")
                break 
            
    writer.close()
    
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write("-" * 50 + "\n")
        f.write("🎉 QUÁ TRÌNH HUẤN LUYỆN ĐÃ KẾT THÚC!\n")
        
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN ĐÃ KẾT THÚC!")
    return model


# ==============================================================================
# KHỐI KHỞI CHẠY THỰC TẾ (EntryPoint)
# ==============================================================================
if __name__ == "__main__":
    import pandas as pd
    import random
    import numpy as np
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Đang sử dụng thiết bị: {device.upper()}")

    print("⏳ Đang import dữ liệu từ ổ đĩa...")
    try:
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH) 
        print(f"✅ Đã nạp thành công: {len(train_df)} câu Train | {len(val_df)} câu Val")
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy dữ liệu! Vui lòng kiểm tra lại đường dẫn trong config.py\nChi tiết: {e}")
        exit()
    
    print(f"🤖 Đang chuẩn bị mô hình: {config.MODEL_TYPE.upper()}...")
    
    if config.MODEL_TYPE == "transformer_lora":
        from source.models.transformer_lora import build_lora_model
        model, tokenizer = build_lora_model(config.active_cfg, device=device)
            
    elif config.MODEL_TYPE == "transformer_full":
        from source.models.transformer_full import build_transformer_full_ft
        model, tokenizer = build_transformer_full_ft(
            model_name=config.active_cfg["model_name"], 
            device=device, 
            use_fast=False
        )
            
    elif config.MODEL_TYPE == "lstm":
        from source.data.tokenize_vocab import SyllableSubwordTokenizer
        from source.models.seq2seq_lstm import Encoder, Decoder, Seq2Seq
        tokenizer = SyllableSubwordTokenizer()
        tokenizer.load(config.VOCAB_PATH)
        cfg = config.active_cfg 
        
        encoder = Encoder(
            vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], pad_idx=config.PAD_IDX
        )
        decoder = Decoder(
            vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"], encoder_dim=cfg["hidden_dim"] * 2, 
            n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            pad_idx=config.PAD_IDX
        )
        model = Seq2Seq(encoder, decoder, device)
        model = model.to(device)
        print(f"✅ Khởi tạo thành công mạng LSTM ({cfg['n_layers']} layers).")

    print("📦 Đang đóng gói dữ liệu vào DataLoader...")
    train_loader = create_dataloader(
        dataframe=train_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=True,
        model_type=config.MODEL_TYPE 
    )
    
    val_loader = create_dataloader(
        dataframe=val_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=False,
        model_type=config.MODEL_TYPE 
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
