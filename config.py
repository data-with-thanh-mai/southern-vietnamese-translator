
import os
SEED = 42
# -------------PATHS & DIRECTORIES -------------
BASE_DIR   = "/content/drive/MyDrive/"
DATA_DIR   = BASE_DIR + "dataset/processed/"
CKPT_DIR   = BASE_DIR + "checkpoints/"
LOG_DIR    = BASE_DIR + "logs/"
FIG_DIR    = BASE_DIR + "figures/"

TRAIN_PATH = DATA_DIR + "train.csv"
VAL_PATH   = DATA_DIR + "val.csv"
TEST_PATH  = DATA_DIR + "test.csv"
VOCAB_PATH = DATA_DIR + "vocab_word_level.json" # tạm

for d in [CKPT_DIR, LOG_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------DATA----------------------------------
MAX_SRC_LEN =  163  # cập nhật từ EDA
MAX_TGT_LEN =  145  # cập nhật từ EDA

# Special token indices — phải khớp với WordLevelTokenizer (tạm)
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


BEAM_SIZE  = 4      # beam search width (1 = greedy)
MAX_DECODE = 50     # số token tối đa khi sinh bản dịch


#  MODEL 1: SEQ2SEQ LSTM (Baseline)
LSTM_CFG = {
    # Kiến trúc
    "vocab_size"  : 8000,   # tạm
    "embed_dim"   : 256,    # chiều embedding
    "hidden_dim"  : 512,    # chiều hidden LSTM
    "n_layers"    : 2,      # số layer xếp chồng
    "dropout"     : 0.3,    # dropout giữa các layer

    # Training
    "batch_size"  : 32,
    "epochs"      : 20,
    "lr"          : 5e-4,   # learning rate AdamW
    "weight_decay": 1e-4,   # L2 regularization
    "clip_grad"   : 1.0,    # gradient clipping

    # Teacher Forcing (giảm tuyến tính theo epoch)
    "tf_start"    : 0.9,    # epoch đầu
    "tf_end"      : 0.5,    # epoch cuối

    # Early Stopping
    "patience"    : 5,
    "min_delta"   : 1e-4,

    # Checkpoint
    "ckpt_path"   : CKPT_DIR + "seq2seq_best.pth",
}


#  MODEL 2: viT5 FULL FINE-TUNING (Performance ceiling) - TẠM

VIT5_CFG = {
    # Pretrained model
    "model_name"  : "VietAI/vit5-base",  # hoặc "vinai/bartpho-word"

    # Kiến trúc (kế thừa từ pretrained, chỉ cần dropout head)
    "dropout"     : ,

    # Training — LR nhỏ hơn nhiều so với LSTM
    # vì fine-tuning: không muốn "phá" pretrained weights
    "batch_size"  :  ,    # nhỏ hơn vì model nặng hơn nhiều
    "epochs"      :  ,
    "lr"          :  # điển hình cho fine-tuning Transformer
    "weight_decay": ,
    "clip_grad"   : ,

    # Scheduler: warmup 10% steps đầu, sau đó linear decay
    "warmup_ratio": ,

    # Early Stopping
    "patience"    : ,      
    "min_delta"   : ,

    # Checkpoint
    "ckpt_path"   : CKPT_DIR + "vit5_full_best.pth",
}


#  MODEL 3: viT5 + LoRA — PEFT (Resource-efficient) - TẠM

LORA_CFG = {
    # Pretrained model (cùng backbone với Model 2)
    "model_name"  : "VietAI/vit5-base",

    # ── LoRA hyperparameters ──────────────────────────────
    # r: thứ hạng ma trận nén — càng nhỏ càng tiết kiệm VRAM
    #    nhưng có thể mất biểu diễn → thường thử r=4,8,16
    "lora_r"      : ,

    # alpha: scaling factor — thường đặt bằng 2*r
    #        điều chỉnh mức độ ảnh hưởng của LoRA lên weight gốc
    #        W_new = W_pretrained + (alpha/r) * B*A
    "lora_alpha"  : ,

    # dropout trong LoRA layers
    "lora_dropout": ,

    # target_modules: áp LoRA vào lớp nào của Transformer
    # q = query, v = value (thường đủ, không cần k và o)
    "target_modules": ["q", "v"],

    # bias: "none" → không train bias → tiết kiệm thêm tham số
    "bias"        : "none",

    # ── Training ─────────────────────────────────────────
    # LR cao hơn Full FT vì LoRA layer được khởi tạo random
    # (không có pretrained weights như Full FT)
    "batch_size"  : ,     # nhiều hơn Full FT vì VRAM thấp hơn
    "epochs"      : ,
    "lr"          : ,
    "weight_decay": ,
    "clip_grad"   : ,

    "warmup_ratio": ,

    # Early Stopping
    "patience"    : ,
    "min_delta"   : ,

    # Checkpoint
    "ckpt_path"   : CKPT_DIR + "vit5_lora_best.pth",
}

# ──────────────────────────────────────────────────────────────
#  ABLATION STUDY — 3 cấu hình LoRA để so sánh
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

def get_tf_ratio(epoch: int, cfg: dict = LSTM_CFG) -> float:
    """
    Teacher Forcing ratio giảm tuyến tính theo epoch.
    Chỉ dùng cho Model 1 (LSTM) — Transformer không cần TF.

    Epoch 0  → tf_start (0.9)   học nhiều từ ground truth
    Epoch 20 → tf_end   (0.5)   tự lực là chính
    """
    start  = cfg["tf_start"]
    end    = cfg["tf_end"]
    epochs = cfg["epochs"]
    ratio  = start - (start - end) * epoch / epochs
    return max(end, ratio)


def get_cfg(model_name: str) -> dict:
    """
    Lấy config theo tên model — dùng trong train.py với argparse.

    Ví dụ:
        cfg = get_cfg("lora")
        model = build_lora_model(cfg)
    """
    mapping = {
        "lstm"  : LSTM_CFG,
        "vit5"  : VIT5_CFG,
        "lora"  : LORA_CFG,
    }
    assert model_name in mapping, \
        f"Model không hợp lệ: {model_name}. Chọn: {list(mapping.keys())}"
    return mapping[model_name]


def print_config(model_name: str = "all") -> None:
    """In config ra console — gọi đầu train.py để log lại."""

    def _print(name, cfg):
        print(f"\n{'─'*45}")
        print(f"  {name}")
        print(f"{'─'*45}")
        for k, v in cfg.items():
            print(f"  {k:<18}: {v}")

    print("=" * 45)
    print("       CẤU HÌNH THỰC NGHIỆM")
    print(f"       SEED = {SEED}")
    print("=" * 45)

    if model_name == "all":
        _print("MODEL 1: Seq2Seq LSTM (Baseline)", LSTM_CFG)
        _print("MODEL 2: viT5 Full Fine-Tuning",   VIT5_CFG)
        _print("MODEL 3: viT5 + LoRA (PEFT)",      LORA_CFG)
    else:
        _print(f"MODEL: {model_name.upper()}", get_cfg(model_name))

    print("=" * 45)



