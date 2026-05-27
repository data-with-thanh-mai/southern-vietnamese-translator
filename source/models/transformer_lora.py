!pip install transformers peft

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. TẢI TOKENIZER VÀ MÔ HÌNH PRE-TRAINED TỪ HUGGING FACE
# Có thể đổi thành các model tiếng Việt như "vinai/bartpho-word" hoặc model nhẹ như "t5-small"
model_name = "VietAI/vit5-base" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Cấu hình LoRA
# Các tham số này khớp 100% với những gì tự code ở hàm __init__
lora_config = LoraConfig(
    r=8,                        # Hạng (Rank) của ma trận LoRA
    lora_alpha=16,              # Hệ số scale
    target_modules=["q", "v"],  # Chỉ gắn LoRA vào Query và Value 
    lora_dropout=0.1,           # Tránh học vẹt
    bias="none",                # Không train thêm bias để tối ưu hoàn toàn
    task_type=TaskType.SEQ_2_SEQ_LM # Bài toán dịch thuật (Sequence-to-Sequence)
)

# 3. Tích hợp LoRA (sổ tay) vào base_model (mô hình gốc)
model = get_peft_model(base_model, lora_config)

# 4. In ra console % tham số cần huấn luyện
print("\n" + "="*50)
print(" KIỂM TRA THÔNG SỐ HUẤN LUYỆN LORA")
print("="*50)
model.print_trainable_parameters()
print("="*50 + "\n")
