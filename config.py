

# -------------PATHS & DIRECTORIES -------------
# Cấu trúc cho Google Colab / Kaggle
import os

SEED = 42

# -------------PATHS & DIRECTORIES (ĐƯỜNG DẪN CỤC BỘ) -------------
# Sử dụng đường dẫn tương đối tính từ thư mục gốc của dự án
DATA_DIR   = "data/processed/"
CKPT_DIR   = "outputs/checkpoints/"
LOG_DIR    = "outputs/logs/"
FIG_DIR    = "outputs/figures/"

# Đường dẫn sẽ tự động nối thành "data/processed/train.csv"
TRAIN_PATH = DATA_DIR + "train(2).csv"  
VAL_PATH   = DATA_DIR + "val.csv"    
TEST_PATH  = DATA_DIR + "test.csv"   
VOCAB_PATH = DATA_DIR + "vocab_word_level.json" 

# Tự động tạo các thư mục lưu checkpoint và log nếu chưa tồn tại
for d in [CKPT_DIR, LOG_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
# -----------------------DATA----------------------------------
MAX_SRC_LEN =  163  
MAX_TGT_LEN =  145  

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

BEAM_SIZE  = 4      
MAX_DECODE = 50     


# ==============================================================
# MODEL 1: SEQ2SEQ LSTM (Baseline)
# ==============================================================
LSTM_CFG = {
    "vocab_size"  : 8000,   
    "embed_dim"   : 256,    
    "hidden_dim"  : 512,    
    "n_layers"    : 2,      
    "dropout"     : 0.3,    
    
    "batch_size"  : 32,
    "epochs"      : 20,
    "lr"          : 5e-4,   
    "weight_decay": 1e-4,   
    "clip_grad"   : 1.0,    
    
    "tf_start"    : 0.9,    
    "tf_end"      : 0.5,    
    
    "patience"    : 5,
    "min_delta"   : 1e-4,
    "ckpt_path"   : CKPT_DIR + "seq2seq_best.pth",
}

# ==============================================================
# MODEL 2: viT5 FULL FINE-TUNING (Performance ceiling)
# ==============================================================
VIT5_CFG = {
    "model_name"  : "VietAI/vit5-base", 
    "dropout"     : 0.1,    # Transformer thường dùng 0.1

    # Training (Cần cẩn thận vì Full FT dễ làm vỡ trọng số pre-trained)
    "batch_size"  : 8,      # Để 8 hoặc 16 tránh tràn RAM (OOM)
    "epochs"      : 10,     # Full FT hội tụ rất nhanh, 10-15 là đủ
    "lr"          : 2e-5,   # LR cực nhỏ để bảo vệ kiến thức gốc (chuẩn LLM)
    "weight_decay": 0.01,   # AdamW chuẩn
    "clip_grad"   : 1.0,

    "warmup_ratio": 0.1,    # 10% steps đầu để hâm nóng
    
    "patience"    : 3,      
    "min_delta"   : 1e-4,
    "ckpt_path"   : CKPT_DIR + "vit5_full_best.pth",
}

# ==============================================================
# MODEL 3: viT5 + LoRA — PEFT (Resource-efficient)
# ==============================================================
LORA_CFG = {
    "model_name"  : "VietAI/vit5-base",
    
    "lora_r"      : 8,      # Rank nén chuẩn
    "lora_alpha"  : 16,     # Thường đặt = 2 * r
    "lora_dropout": 0.05,   
    "target_modules": ["q", "v"], 
    "bias"        : "none",

    # Training (LoRA layer khởi tạo ngẫu nhiên nên cần LR to hơn Full FT)
    "batch_size"  : 16,     # LoRA siêu nhẹ, có thể nhét batch to hơn
    "epochs"      : 15,     # Cần thời gian học lâu hơn Full FT một chút
    "lr"          : 1e-4,   # LR lớn hơn (1e-4) là tiêu chuẩn vàng của LoRA
    "weight_decay": 0.01,
    "clip_grad"   : 1.0,

    "warmup_ratio": 0.1,
    
    "patience"    : 3,
    "min_delta"   : 1e-4,
    "ckpt_path"   : CKPT_DIR + "vit5_lora_best.pth",
}

# ──────────────────────────────────────────────────────────────
# ABLATION STUDY
# ──────────────────────────────────────────────────────────────
LORA_ABLATION = {
    "lora_v1": {**LORA_CFG, "lora_r": 4,  "lora_alpha": 8,
                "ckpt_path": CKPT_DIR + "lora_r4_best.pth"},   
    "lora_v2": {**LORA_CFG, "lora_r": 8,  "lora_alpha": 16,
                "ckpt_path": CKPT_DIR + "lora_r8_best.pth"},   
    "lora_v3": {**LORA_CFG, "lora_r": 16, "lora_alpha": 32,
                "ckpt_path": CKPT_DIR + "lora_r16_best.pth"},  
}

#-------MỘT SỐ HÀM PHỤ-----------------------
def get_cfg(model_name: str) -> dict:
    mapping = {
        "lstm"  : LSTM_CFG,
        "vit5"  : VIT5_CFG,
        "lora"  : LORA_CFG,
    }
    assert model_name in mapping, \
        f"Model không hợp lệ: {model_name}. Chọn: {list(mapping.keys())}"
    return mapping[model_name]

# ==============================================================
# TRẠM TRUNG CHUYỂN (ÉP KHỚP VỚI TRAIN.PY)
# ==============================================================
# BẠN ĐỔI CHỮ BÊN DƯỚI THÀNH "lstm", "vit5", HOẶC "lora" ĐỂ CHẠY
CHOSEN_MODEL = "lstm"

active_cfg = get_cfg(CHOSEN_MODEL)

# 1. Khớp biến MODEL_TYPE cho file train.py hiểu
if CHOSEN_MODEL == "lstm":
    MODEL_TYPE = "lstm"
elif CHOSEN_MODEL == "vit5":
    MODEL_TYPE = "transformer_full"
elif CHOSEN_MODEL == "lora":
    MODEL_TYPE = "transformer_lora"

# 2. Bung các biến trong Dictionary ra ngoài Global
LEARNING_RATE = active_cfg["lr"]
NUM_EPOCHS    = active_cfg["epochs"]
BATCH_SIZE    = active_cfg["batch_size"]
WEIGHT_DECAY  = active_cfg.get("weight_decay", 1e-4)
WARMUP_RATIO  = active_cfg.get("warmup_ratio", 0.1)
PATIENCE      = active_cfg.get("patience", 3)
TF_RATIO      = active_cfg.get("tf_end", 0.5)
