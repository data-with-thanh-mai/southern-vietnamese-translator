import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Metrics ──────────────────────────────────────────────────────────────────
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
from sentence_transformers import SentenceTransformer, util

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

import config
from source.baselines.rule_based import RuleBasedTranslator


# ==============================================================================
# 1. HÀM SINH BẢN DỊCH CHO TỪNG MODEL
# ==============================================================================

def predict_lstm(model, tokenizer, texts: list[str], device: str) -> list[str]:
    """Dùng translate_greedy của Seq2Seq LSTM."""
    model.eval()
    predictions = []

    for text in tqdm(texts, desc="  LSTM đang dịch"):
        token_ids = tokenizer.encode(text)
        src = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            pred_ids = model.translate_greedy(
                src=src,
                pad_idx=config.PAD_IDX,
                bos_idx=config.BOS_IDX,
                eos_idx=config.EOS_IDX,
                max_len=config.MAX_DECODE,
            )

        pred_text = tokenizer.decode(pred_ids)  
        predictions.append(pred_text)

    return predictions


def predict_transformer(model, tokenizer, texts: list[str], device: str) -> list[str]:
    """Dùng model.generate() cho viT5 full / LoRA."""
    model.eval()
    predictions = []

    for text in tqdm(texts, desc="  Transformer đang dịch"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SRC_LEN,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.MAX_DECODE,
                num_beams=config.BEAM_SIZE,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_text)

    return predictions


def predict_rule_based(train_df: pd.DataFrame, texts: list[str]) -> list[str]:
    """Dùng RuleBasedTranslator."""
    translator = RuleBasedTranslator(train_df)
    predictions = []
    for text in tqdm(texts, desc="  Rule-based đang dịch"):
        predictions.append(translator.translate_sentence(text))
    return predictions


# ==============================================================================
# 2. TÍNH METRICS
# ==============================================================================

def compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    """corpus BLEU-4 với smoothing."""
    refs_tokenized = [[str(ref).split()] for ref in references]
    hyps_tokenized = [str(hyp).split() for hyp in hypotheses]
    smoother = SmoothingFunction().method1
    return corpus_bleu(refs_tokenized, hyps_tokenized, smoothing_function=smoother) * 100


def compute_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(str(ref), str(hyp))["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return np.mean(scores) * 100


def compute_meteor(references: list[str], hypotheses: list[str]) -> float:
    scores = [
        meteor_score([str(ref).split()], str(hyp).split())
        for ref, hyp in zip(references, hypotheses)
    ]
    return np.mean(scores) * 100


def compute_chrf(references: list[str], hypotheses: list[str]) -> float:
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return result.score


def compute_semantic_similarity(
    references: list[str], hypotheses: list[str], sem_model
) -> float:
    ref_embs = sem_model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    hyp_embs = sem_model.encode(hypotheses, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(ref_embs, hyp_embs).diagonal()
    return cosine_scores.mean().item() * 100


def evaluate_predictions(
    references: list[str],
    hypotheses: list[str],
    sem_model,
    model_name: str,
) -> dict:
    print(f"\n Tính metrics cho [{model_name}]...")
    return {
        "Model"              : model_name,
        "BLEU"               : round(compute_bleu(references, hypotheses), 2),
        "ROUGE-L"            : round(compute_rouge_l(references, hypotheses), 2),
        "METEOR"             : round(compute_meteor(references, hypotheses), 2),
        "chrF"               : round(compute_chrf(references, hypotheses), 2),
        "Semantic Similarity": round(compute_semantic_similarity(references, hypotheses, sem_model), 2),
    }


# ==============================================================================
# 3. LOAD CHECKPOINT
# ==============================================================================

def load_lstm(device: str):
    from source.data.tokenize_vocab import SyllableSubwordTokenizer
    from source.models.seq2seq_lstm import Encoder, Decoder, Seq2Seq

    tokenizer = SyllableSubwordTokenizer()
    tokenizer.load(config.VOCAB_PATH)

    cfg = config.LSTM_CFG
    encoder = Encoder(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
        dropout=0.0, pad_idx=config.PAD_IDX,
    )
    decoder = Decoder(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], encoder_dim=cfg["hidden_dim"] * 2,
        n_layers=cfg["n_layers"], dropout=0.0, pad_idx=config.PAD_IDX,
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    ckpt_path = os.path.join("outputs", "checkpoints", "best_model_lstm", "lstm_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint LSTM tại: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Đã load LSTM từ {ckpt_path}")
    return model, tokenizer


def load_transformer_full(device: str):
    from source.models.transformer_full import build_transformer_full_ft

    ckpt_path = os.path.join("outputs", "checkpoints", "best_model_transformer_full")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint Transformer Full tại: {ckpt_path}")

    model, tokenizer = build_transformer_full_ft(
        model_name=ckpt_path, device=device, use_fast=False
    )
    print(f"Đã load Transformer Full từ {ckpt_path}")
    return model, tokenizer


def load_transformer_lora(device: str):
    from source.models.transformer_lora import build_lora_model

    ckpt_path = os.path.join("outputs", "checkpoints", "best_model_transformer_lora")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint LoRA tại: {ckpt_path}")

    model, tokenizer = build_lora_model(config.LORA_CFG, device=device, checkpoint_path=ckpt_path)
    print(f" Đã load Transformer LoRA từ {ckpt_path}")
    return model, tokenizer


# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nThiết bị: {device.upper()}")

    print("Đang load tập test...")
    test_df  = pd.read_csv(config.TEST_PATH)
    train_df = pd.read_csv(config.TRAIN_PATH)

    sources    = test_df["source_sentence"].astype(str).tolist()
    references = test_df["target_sentence"].astype(str).tolist()
    print(f" {len(test_df)} câu test")

    print("\n🔍 Đang load mô hình Semantic Similarity...")
    sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    all_results = []
    all_preds   = {}

    # ── MODEL 1: LSTM ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(" MODEL 1: LSTM")
    print("="*60)
    try:
        lstm_model, lstm_tok = load_lstm(device)
        preds_lstm = predict_lstm(lstm_model, lstm_tok, sources, device)
        all_preds["LSTM"] = preds_lstm
        all_results.append(evaluate_predictions(references, preds_lstm, sem_model, "LSTM"))
    except FileNotFoundError as e:
        print(f" Bỏ qua LSTM: {e}")

    # ── MODEL 2: Transformer Full ──────────────────────────────────
    print("\n" + "="*60)
    print("MODEL 2: Transformer Full Fine-tuning")
    print("="*60)
    try:
        full_model, full_tok = load_transformer_full(device)
        preds_full = predict_transformer(full_model, full_tok, sources, device)
        all_preds["Transformer Full"] = preds_full
        all_results.append(evaluate_predictions(references, preds_full, sem_model, "Transformer Full"))
        del full_model
    except FileNotFoundError as e:
        print(f"  ⚠️  Bỏ qua Transformer Full: {e}")

    # ── MODEL 3: Transformer LoRA ──────────────────────────────────
    print("\n" + "="*60)
    print("MODEL 3: Transformer LoRA")
    print("="*60)
    try:
        lora_model, lora_tok = load_transformer_lora(device)
        preds_lora = predict_transformer(lora_model, lora_tok, sources, device)
        all_preds["Transformer LoRA"] = preds_lora
        all_results.append(evaluate_predictions(references, preds_lora, sem_model, "Transformer LoRA"))
        del lora_model
    except FileNotFoundError as e:
        print(f"Bỏ qua Transformer LoRA: {e}")

    # ── BASELINE: Rule-based ───────────────────────────────────────
    print("\n" + "="*60)
    print("BASELINE: Rule-based")
    print("="*60)
    preds_rule = predict_rule_based(train_df, sources)
    all_preds["Rule-based"] = preds_rule
    all_results.append(evaluate_predictions(references, preds_rule, sem_model, "Rule-based"))

    # ── BẢNG KẾT QUẢ ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("BẢNG KẾT QUẢ TỔNG HỢP")
    print("="*60)
    results_df = pd.DataFrame(all_results).set_index("Model")
    print(results_df.to_string())

    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/evaluation_results.csv"
    results_df.to_csv(csv_path)
    print(f"\nĐã lưu bảng kết quả vào: {csv_path}")

    # ── VÍ DỤ DỊCH THỬ ────────────────────────────────────────────
    print("\n" + "="*60)
    print("VÍ DỤ DỊCH THỬ (5 câu đầu)")
    print("="*60)
    for i in range(min(5, len(sources))):
        print(f"\n[Câu {i+1}]")
        print(f"Input     : {sources[i]}")
        print(f"Reference : {references[i]}")
        for model_name, preds in all_preds.items():
            print(f"{model_name:20s}: {preds[i]}")

    examples = []
    for i in range(len(sources)):
        row = {"input": sources[i], "reference": references[i]}
        for model_name, preds in all_preds.items():
            row[model_name] = preds[i]
        examples.append(row)

    json_path = "outputs/evaluation_examples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu ví dụ dịch vào: {json_path}")
    print("\nĐÁNH GIÁ HOÀN TẤT!")


if __name__ == "__main__":
    main()


# ==============================================================================
# DEMO: TỰ NHẬP CÂU ĐỂ DỊCH THỬ
# ==============================================================================

def demo_interactive(all_preds_models: dict, device: str):
    """
    all_preds_models: dict chứa model/tokenizer đã load sẵn
    {
      "LSTM": (model, tokenizer),
      "Rule-based": (train_df,),
      ...
    }
    """
    print("\n" + "="*60)
    print("🎤 CHẾ ĐỘ DỊCH THỬ — TỰ NHẬP CÂU")
    print("="*60)
    print("Nhập câu phương ngữ miền Nam, gõ 'quit' để thoát.\n")

    while True:
        user_input = input("📝 Nhập câu: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Thoát chế độ dịch thử.")
            break
        if not user_input:
            continue

        print()
        for model_name, pack in all_preds_models.items():
            if model_name == "LSTM":
                model, tokenizer = pack
                preds = predict_lstm(model, tokenizer, [user_input], device)
                print(f"  🔷 LSTM             : {preds[0]}")

            elif model_name == "Transformer Full":
                model, tokenizer = pack
                preds = predict_transformer(model, tokenizer, [user_input], device)
                print(f"  🔷 Transformer Full : {preds[0]}")

            elif model_name == "Transformer LoRA":
                model, tokenizer = pack
                preds = predict_transformer(model, tokenizer, [user_input], device)
                print(f"  🔷 Transformer LoRA : {preds[0]}")

            elif model_name == "Rule-based":
                train_df = pack[0]
                preds = predict_rule_based(train_df, [user_input])
                print(f"  🔷 Rule-based       : {preds[0]}")
        print()


if __name__ == "__main__":
    import sys
    # Nếu chạy với flag --demo thì chỉ chạy demo, bỏ qua evaluate
    if "--demo" in sys.argv:
        import pandas as pd
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Thiết bị: {device.upper()}")

        train_df = pd.read_csv(config.TRAIN_PATH)
        loaded_models = {}

        # Load LSTM
        try:
            lstm_model, lstm_tok = load_lstm(device)
            loaded_models["LSTM"] = (lstm_model, lstm_tok)
        except FileNotFoundError as e:
            print(f"⚠️  Bỏ qua LSTM: {e}")

        # Load Transformer Full
        try:
            full_model, full_tok = load_transformer_full(device)
            loaded_models["Transformer Full"] = (full_model, full_tok)
        except FileNotFoundError as e:
            print(f"⚠️  Bỏ qua Transformer Full: {e}")

        # Load Transformer LoRA
        try:
            lora_model, lora_tok = load_transformer_lora(device)
            loaded_models["Transformer LoRA"] = (lora_model, lora_tok)
        except FileNotFoundError as e:
            print(f"⚠️  Bỏ qua Transformer LoRA: {e}")

        # Rule-based luôn có
        loaded_models["Rule-based"] = (train_df,)

        demo_interactive(loaded_models, device)
    else:
        main()
