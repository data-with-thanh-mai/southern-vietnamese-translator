# -------------PATHS & DIRECTORIES -------------
import os

SEED = 42

# Sử dụng đường dẫn tương đối tính từ thư mục gốc của dự án
DATA_DIR   = "data/processed/"
CKPT_DIR   = "outputs/checkpoints/"
LOG_DIR    = "outputs/logs/"
FIG_DIR    = "outputs/figures/"

TRAIN_PATH = DATA_DIR + "train (2).csv"  # Đã sửa lại cho chuẩn
VAL_PATH   = DATA_DIR + "test.csv"    
TEST_PATH  = DATA_DIR + "test.csv"   
VOCAB_PATH = DATA_DIR + "vocab_word_level.json" 

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
    "pad_idx"     : PAD_IDX
}

# ==============================================================
# MODEL 2: viT5 FULL FINE-TUNING (Performance ceiling)
# ==============================================================
VIT5_CFG = {
    "model_name"  : "VietAI/vit5-base", 
    "dropout"     : 0.1,    

    "batch_size"  : 8,      
    "epochs"      : 10,     
    "lr"          : 2e-5,   
    "weight_decay": 0.01,   
    "clip_grad"   : 1.0,
    "warmup_ratio": 0.1,    
    
    "patience"    : 3,      
    "min_delta"   : 1e-4,
    "ckpt_path"   : CKPT_DIR + "vit5_full_best.pth",
}

# ==============================================================
# MODEL 3: viT5 + LoRA — PEFT (Resource-efficient)
# ==============================================================
LORA_CFG = {
    "model_name"  : "VietAI/vit5-base",
    
    "lora_r"      : 8,      
    "lora_alpha"  : 16,     
    "lora_dropout": 0.05,   
    "target_modules": ["q", "v"], 
    "bias"        : "none",

    "batch_size"  : 16,     
    "epochs"      : 15,     
    "lr"          : 1e-4,   
    "weight_decay": 0.01,
    "clip_grad"   : 1.0,
    "warmup_ratio": 0.1,
    
    "patience"    : 3,
    "min_delta"   : 1e-4,
    "ckpt_path"   : CKPT_DIR + "vit5_lora_best.pth",
}

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
# TRẠM TRUNG CHUYỂN
# ==============================================================
CHOSEN_MODEL = "vit5" # Đổi thành "lstm" hoặc "vit5" nếu muốn

active_cfg = get_cfg(CHOSEN_MODEL)

if CHOSEN_MODEL == "lstm":
    MODEL_TYPE = "lstm"
elif CHOSEN_MODEL == "vit5":
    MODEL_TYPE = "transformer_full"
elif CHOSEN_MODEL == "lora":
    MODEL_TYPE = "transformer_lora"

LEARNING_RATE = active_cfg["lr"]
NUM_EPOCHS    = active_cfg["epochs"]
BATCH_SIZE    = active_cfg["batch_size"]
WEIGHT_DECAY  = active_cfg.get("weight_decay", 1e-4)
WARMUP_RATIO  = active_cfg.get("warmup_ratio", 0.1)
PATIENCE      = active_cfg.get("patience", 3)
TF_RATIO      = active_cfg.get("tf_end", 0.5)
