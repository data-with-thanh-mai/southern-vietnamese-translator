import torch
import re
from underthesea import sent_tokenize

def translate_text(model, tokenizer, source_text, device="cpu", 
                   max_source_len=256, max_target_len=64, num_beams=5):
    """
    Hàm sinh văn bản đã tách biệt rõ ràng giới hạn đầu vào và đầu ra.
    """
    model.eval()
    
    # 1. GIỚI HẠN ĐẦU VÀO
    inputs = tokenizer(
        source_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_len  
    ).to(device)
    
    # 2. GIỚI HẠN ĐẦU RA
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_target_len, 
            num_beams=num_beams, 
            early_stopping=True if num_beams > 1 else False,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_text

# ==========================================
# CÁCH 1: TÁCH CÂU BẰNG REGEX (CƠ BẢN)
# ==========================================
def translate_paragraph_regex(model, tokenizer, paragraph, device="cpu", 
                              max_source_len=256, max_target_len=64, num_beams=5):
    """
    Tách đoạn văn bằng Regex (Dựa vào dấu . ! ?).
    Nhược điểm: Dễ cắt sai ở các từ viết tắt như TP., Ths.
    """
    sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())
    translated_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        trans_sent = translate_text(
            model=model, tokenizer=tokenizer, source_text=sentence, 
            device=device, max_source_len=max_source_len, 
            max_target_len=max_target_len, num_beams=num_beams
        )
        translated_sentences.append(trans_sent)
        
    return " ".join(translated_sentences)

# ==========================================
# CÁCH 2: TÁCH CÂU BẰNG AI (NÂNG CẤP)
# ==========================================
def translate_paragraph_underthesea(model, tokenizer, paragraph, device="cpu", 
                                    max_source_len=256, max_target_len=64, num_beams=5):
    """
    Tách đoạn văn bằng AI của Underthesea.
    Ưu điểm: Hiểu ngữ cảnh tiếng Việt, không bị lỗi ở các từ viết tắt.
    """
    sentences = sent_tokenize(paragraph)
    translated_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        trans_sent = translate_text(
            model=model, tokenizer=tokenizer, source_text=sentence, 
            device=device, max_source_len=max_source_len, 
            max_target_len=max_target_len, num_beams=num_beams
        )
        translated_sentences.append(trans_sent)
        
    return " ".join(translated_sentences)
