
import argparse
import time
import math
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from config     import *
from utils      import set_seed, EarlyStopping, save_checkpoint, load_checkpoint
from data_loader import get_dataloaders, WordLevelTokenizer
from seq2seq_lstm import Seq2Seq



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
            "weight_decay": cfg.get("weight_decay", 1e-4),
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
        lr=cfg["lr"],
        betas=(0.9,0.999),   # beta mặc định nên giữ nguyên
        eps=1e-8 #sai số
    )
    # ---Linear Warmup + Linear Decay Scheduler ------
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))

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
    print(f"[Optimizer] AdamW | lr={cfg['lr']} | "
          f"weight_decay={cfg.get('weight_decay', 1e-4)}")
    print(f"[Scheduler] Linear Warmup + Decay | "
          f"warmup={warmup_steps}/{total_steps} steps "
          f"({cfg.get('warmup_ratio', 0.1)*100:.0f}%)")

    return optimizer, scheduler

#### Phần Lưu checkpoint, đánh giá loss của epoch, nhận vô optimizer và model, val loader và train loader, cơ chế Early stopping
import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, optimizer, epochs=15, patience=3, device="cuda"):
    print(" BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")
    
    best_val_loss = float('inf')
    
    # Biến đếm số lần Loss không giảm
    epochs_no_improve = 0 

    history_train_loss = []
    history_val_loss = []
    
    for epoch in range(epochs):
        # ==========================================
        # 1. PHA HUẤN LUYỆN (TRAINING)
        # ==========================================
        model.train() 
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # 2. PHA ĐÁNH GIÁ (VALIDATION)
        # ==========================================
        model.eval() 
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad(): 
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['target_ids'].to(device)
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # ==========================================
        # 3. LOG, LƯU CHECKPOINT VÀ EARLY STOPPING
        # ==========================================
        print(f" Kết quả Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Ghi kết quả để vẽ hình
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"    KỶ LỤC MỚI! Val Loss giảm từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            
            # Reset lại bộ đếm về 0
            epochs_no_improve = 0 
            
            # ĐỊNH NGHĨA ĐƯỜNG DẪN THEO CẤU TRÚC THƯ MỤC CỦA BẠN
            checkpoint_dir = os.path.join("outputs", "checkpoints")
            
            # Đảm bảo thư mục tồn tại trước khi lưu
            os.makedirs(checkpoint_dir, exist_ok=True) 
            
            # LƯU TRỌNG SỐ LORA BẰNG HÀM CỦA HUGGING FACE
            # Hàm save_pretrained sẽ tự tạo các file như adapter_model.bin/safetensors và adapter_config.json
            model.save_pretrained(checkpoint_dir)
            print(f"    Đã lưu thành công mô hình tốt nhất vào: {checkpoint_dir}")
            
        else:
            # Tăng bộ đếm nếu mô hình không tiến bộ
            epochs_no_improve += 1
            print(f"    Val Loss không giảm. Cảnh báo Overfitting lần {epochs_no_improve}/{patience}.")
            
            # Kiểm tra xem đã hết kiên nhẫn chưa
            if epochs_no_improve >= patience:
                print(f" ĐÃ KÍCH HOẠT EARLY STOPPING! Dừng huấn luyện sớm ở Epoch {epoch+1} để tránh học vẹt.")
                break # Phá vỡ vòng lặp for, kết thúc train luôn!
            
    print(" QUÁ TRÌNH HUẤN LUYỆN ĐÃ KẾT THÚC!")


