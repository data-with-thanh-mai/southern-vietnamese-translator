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

# OPTIMIZER & LR SCHEDULER
# AdamW = Adam + Weight Decay tách biệt
#   Adam thông thường: weight decay bị "lẫn" vào adaptive gradient
#   AdamW: áp weight decay trực tiếp lên weight --> regularize đúng hơn
#
#
# Linear Warmup + Decay:
#   Epoch 0 --> warmup_steps: LR tăng từ 0 --> lr_max (tránh diverge lúc đầu)
#   warmup_steps --> cuối: LR giảm tuyến tính về 0  (hội tụ mượt)
# Giair thích warmup:
#   Lúc đầu model weights random --> gradient lớn, không ổn định
#   Nếu LR cao ngay từ đầu --> bước nhảy quá lớn --> diverge
#   Warmup: "hâm nóng" từ từ trước khi chạy full lr

def build_optimizer_and_scheduler(
    model        : nn.Module,
    cfg          : dict,
    total_steps  : int
) -> tuple:
    """
    Khởi tạo AdamW và Linear Warmup Scheduler.

    Args:
        models      : Seq2Seq...
        cfg        : config dict (LSTM_CFG hoặc LORA_CFG)
        total_steps: tổng số bước train = n_epochs * n_batchesz
    Returns:
        optimizer, scheduler
    """
    # ── AdamW ────────────────────────────────────────────────
    # Tách params có weight decay và không có
    # Bias và LayerNorm KHÔNG nên bị weight decay
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            # Params có weight decay: Linear weights, Embedding, LSTM weights
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": getattr(cfg, "WEIGHT_DECAY", 1e-4),
        },
        {
            # Params không có weight decay: bias, layer norm
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    """ Giai thích lí do tại sao có param groups:
    - Bias và LayerNorm không nên bị weight decay vì:
      + Bias: thường có giá trị nhỏ, nếu decay sẽ bị ảnh hưởng nặng nề
      + LayerNorm: có vai trò ổn định hóa activations, nếu decay sẽ làm mất ổn định
    - Các params còn lại (Linear weights, Embedding, LSTM weights) nên bị decay để regularize
    - Việc tách param groups giúp áp weight decay chính xác theo từng loại param
    """
    optimizer = AdamW(
        param_groups,
        lr=cfg.LEARNING_RATE,
        betas=(0.9,0.999),   # beta mặc định nên giữ nguyên
        eps=1e-8 #sai số
    )
    # ---Linear Warmup + Linear Decay Scheduler ------
    warmup_steps = int(total_steps * getattr(cfg, "WARMUP_RATIO", 0.1))

    def lr_lambda(current_step: int) -> float:
        """
        Hàm nhân LR tại mỗi step:
          Phase warmup : LR tăng tuyến tính 0 → 1.0
          Phase decay  : LR giảm tuyến tính 1.0 → 0.0

        Scheduler nhân lr_lambda(step) * lr_max → LR thực tế
        """
        # Cách tính số lr_lambda(step)
        if current_step < warmup_steps:
            # Warmup: tăng từ 0 --> lr_max
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Decay: giảm từ lr_max --> 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Log thông tin scheduler
    print(f"[Optimizer] AdamW | lr={cfg.LEARNING_RATE} | "
          f"weight_decay={getattr(cfg, 'WEIGHT_DECAY', 1e-4)}")
    print(f"[Scheduler] Linear Warmup + Decay | "
          f"warmup={warmup_steps}/{total_steps} steps "
          f"({getattr(cfg, 'WARMUP_RATIO', 0.1)*100:.0f}%)")

    return optimizer, scheduler



def train_model(model, train_loader, val_loader, tokenizer, config, device="cuda"):
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN.")
    
    # MLOps Setup
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Biến Early Stopping
    best_val_loss = float('inf')
    # Biến đếm số lần Loss không giảm (Comment của Ngọc)
    epochs_no_improve = 0 
    
    # Gọi optimizer và scheduler 
    total_steps = len(train_loader) * config.NUM_EPOCHS
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    # Khởi tạo hàm loss cho lstm- bỏ qua padding 
    if config.MODEL_TYPE == "lstm":
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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
                outputs = model(src=input_ids, trg=labels, teacher_forcing_ratio=config.TF_RATIO)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            loss.backward()
            # Dùng clip-norm 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Ghi log Tensorboard
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
                    # Tắt teacher forcing 
                    outputs = model(src=input_ids, trg=labels, teacher_forcing_ratio=0.0)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch)
        
        # ==========================================
        # 3. LOG, LƯU CHECKPOINT VÀ EARLY STOPPING 
        # ==========================================
        print(f"Kết quả Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"KỶ LỤC MỚI! Val Loss giảm từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            
            # Reset lại bộ đếm về 0 
            epochs_no_improve = 0 
            
            # ĐỊNH NGHĨA ĐƯỜNG DẪN THEO CẤU TRÚC THƯ MỤC CỦA BẠN 
            checkpoint_dir = os.path.join("outputs", "checkpoints", "best_model")
            
            # Đảm bảo thư mục tồn tại trước khi lưu 
            os.makedirs(checkpoint_dir, exist_ok=True) 
            
            # LƯU TRỌNG SỐ LORA BẰNG HÀM CỦA HUGGING FACE 
            # Hàm save_pretrained sẽ tự tạo các file như adapter_model.bin/safetensors và adapter_config.json
            if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir) 
            elif config.MODEL_TYPE == "lstm":
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "lstm_best.pth"))
                
            print(f"Đã lưu thành công mô hình tốt nhất vào: {checkpoint_dir}")
            
        else:
            # Tăng bộ đếm nếu mô hình không tiến bộ 
            epochs_no_improve += 1
            print(f"Val Loss không giảm. Cảnh báo Overfitting lần {epochs_no_improve}/{config.PATIENCE}.")
            
            # Kiểm tra xem đã hết kiên nhẫn chưa 
            if epochs_no_improve >= config.PATIENCE:
                print(f"ĐÃ KÍCH HOẠT EARLY STOPPING! Dừng huấn luyện sớm ở Epoch {epoch+1} để tránh học vẹt.")
                break # Phá vỡ vòng lặp for, kết thúc train luôn! 
            
    writer.close()
    print("QUÁ TRÌNH HUẤN LUYỆN ĐĐ KẾT THÚC!")
    return model

# ==============================================================================
# KHỐI KHỞI CHẠY THỰC TẾ (EntryPoint)
# Thêm vào dưới cùng file train.py
# ==============================================================================
if __name__ == "__main__":
    import pandas as pd
    
    # Cố định seed để kết quả ổn định
    
    
    # Chọn thiết bị (Bật GPU trên Kaggle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Đang sử dụng thiết bị: {device.upper()}")

    # 1. IMPORT DATASET TỪ FILE ĐÃ CHIA (Của Minh)
    print("⏳ Đang import dữ liệu từ ổ đĩa...")
    try:
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.TEST_PATH)
        print(f"✅ Đã nạp thành công: {len(train_df)} câu Train | {len(val_df)} câu Val")
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy dữ liệu! Vui lòng kiểm tra lại đường dẫn trong config.py\nChi tiết: {e}")
        exit()
    
    # 2. KHỞI TẠO TOKENIZER & MODEL TƯƠNG ỨNG VỚI CẤU HÌNH
    print(f"🤖 Đang chuẩn bị mô hình: {config.MODEL_TYPE.upper()}...")
    
    if config.MODEL_TYPE in ["transformer_full", "transformer_lora"]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Tải bộ Tokenizer của viT5
        tokenizer = AutoTokenizer.from_pretrained(config.active_cfg["model_name"])
        # Tải mạng neural của viT5
        model = AutoModelForSeq2SeqLM.from_pretrained(config.active_cfg["model_name"])
        
        # Nếu chọn LoRA (Của Ngọc)
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
            # In ra màn hình xem nén được bao nhiêu tham số
            model.print_trainable_parameters() 
            
    elif config.MODEL_TYPE == "lstm":
        from source.models.seq2seq_lstm import Encoder, Decoder
    
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
            encoder_dim = cfg["hidden_dim"] * 2,  # Bi-LSTM nên x2
            n_layers    = cfg["n_layers"],
            dropout     = cfg["dropout"],
            pad_idx     = cfg["pad_idx"]
        )
        model = Seq2Seq(encoder, decoder, device)
        print(f"✅ Khởi tạo thành công mạng LSTM ({cfg['n_layers']} layers).")
        
    # Nạp mô hình lên Card Đồ Họa
    model = model.to(device)

    # 3. CHUYỂN DATA THÀNH DATALOADER CHO Pytorch
    print("📦 Đang đóng gói dữ liệu vào DataLoader...")
    train_loader = create_dataloader(
        dataframe=train_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=True 
    )
    
    val_loader = create_dataloader(
        dataframe=val_df,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        max_source_len=config.MAX_SRC_LEN,
        max_target_len=config.MAX_TGT_LEN,
        is_train=False 
    )

    # 4. KÍCH HOẠT VÒNG LẶP HUẤN LUYỆN
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
