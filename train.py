
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
#   AdamW: áp weight decay trực tiếp lên weight → regularize đúng hơn
#
#
# Linear Warmup + Decay:
#   Epoch 0 --> warmup_steps: LR tăng từ 0 --> lr_max (tránh diverge lúc đầu)
#   warmup_steps --> cuối: LR giảm tuyến tính về 0  (hội tụ mượt)
# Giair thích warmup:
#   Lúc đầu model weights random --> gradient lớn, không ổn định
#   Nếu LR cao ngay từ đầu --> bước nhảy quá lớn → diverge
#   Warmup: "hâm nóng" từ từ trước khi chạy full speed

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

