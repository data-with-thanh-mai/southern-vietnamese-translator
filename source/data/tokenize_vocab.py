import re
import json
import unicodedata

# ==============================================================================
# HÀM HỖ TRỢ DECODE
# ==============================================================================

def remove_tone(s):
    """Bỏ dấu thanh để so sánh phụ âm đầu."""
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r'[\u0300-\u036f]', '', s)
    s = s.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize("NFC", s).lower()

PHU_AM_DAU_SET = {
    "ngh","ch","gh","kh","nh","ng","ph","th","tr","gi","qu",
    "b","c","d","đ","g","h","k","l","m","n","p","r","s","t","v","x"
}

def merge_subwords(tokens: list) -> str:
    """
    Ghép subword lại thành câu hoàn chỉnh.
    Logic: phụ âm đầu (b, c, t, ch, nh, ...) luôn dính vào token tiếp theo.
    """
    if not tokens:
        return ""
    result = []
    i = 0
    while i < len(tokens):
        tok = str(tokens[i]).strip()
        if not tok:
            i += 1
            continue
        base = remove_tone(tok)
        if base in PHU_AM_DAU_SET and i + 1 < len(tokens):
            next_tok = str(tokens[i + 1]).strip()
            result.append(tok + next_tok)
            i += 2
        else:
            result.append(tok)
            i += 1
    return " ".join(result)


# ==============================================================================
# TOKENIZER CHÍNH
# ==============================================================================

class SyllableSubwordTokenizer:
    def __init__(self):
        self.PHU_AM_DAU = r'^(ch|gh|kh|nh|ngh|ng|ph|th|tr|gi|qu|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)'
        self.NGUYEN_AM  = r'(uyê|uyê|yêu|oai|oao|oay|oeo|uai|uâo|uây|ươi|ươu|uôi|ai|ao|au|âu|ay|eo|ia|iê|io|iu|oa|oă|oâ|oe|oi|ôi|ơi|oo|ôô|ua|uâ|uô|uơ|uy|ưa|ưi|ươ|ưu|vê|a|ă|â|e|ê|i|o|ô|ơ|u|ư|y)'
        self.AM_CUOI    = r'(ch|ng|nh|p|t|c|m|n)$'

        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3
        }

        self.vocab       = {}
        self.id_to_vocab = {}

    def _split_syllable(self, syllable):
        syl_clean = str(syllable).lower().strip()
        p_dau, ng_am, a_cuoi = "", "", ""

        match_dau = re.match(self.PHU_AM_DAU, syl_clean)
        if match_dau:
            p_dau = match_dau.group(1)
            syl_clean = syl_clean[len(p_dau):]

        match_cuoi = re.search(self.AM_CUOI, syl_clean)
        if match_cuoi:
            a_cuoi = match_cuoi.group(1)
            syl_clean = syl_clean[:-len(a_cuoi)]

        ng_am = syl_clean
        return p_dau, ng_am, a_cuoi

    def _extract_components(self, word):
        p_dau, ng_am, a_cuoi = self._split_syllable(word)
        components = set()
        if p_dau: components.add(p_dau)
        if ng_am: components.add(ng_am)
        if a_cuoi: components.add(a_cuoi)
        if ng_am and a_cuoi: components.add(ng_am + a_cuoi)
        components.add(word)
        for char in word: components.add(char)
        return components

    def fit(self, word_list):
        self.vocab = self.special_tokens.copy()
        all_sub_tokens = set()
        for item in word_list:
            text = str(item).lower()
            clean_item = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
            for word in clean_item.split():
                all_sub_tokens.update(self._extract_components(word))

        current_id = len(self.vocab)
        for token in sorted(all_sub_tokens):
            if token not in self.vocab:
                self.vocab[token] = current_id
                current_id += 1

        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tổng số lượng từ khóa trong từ điển: {len(self.vocab)}")

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu từ điển vào file: {filepath}")

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Đã tải thành công từ điển với {len(self.vocab)} token.")

    def get_vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        clean_text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', str(text).lower())
        words = clean_text.split()
        final_tokens = ["[CLS]"]
        final_ids    = [self.vocab["[CLS]"]]

        for word in words:
            p_dau, ng_am, a_cuoi = self._split_syllable(word)
            tokens = []
            if p_dau in self.vocab: tokens.append(p_dau)
            if (ng_am + a_cuoi) in self.vocab:
                tokens.append(ng_am + a_cuoi)
            elif ng_am in self.vocab:
                tokens.append(ng_am)
                if a_cuoi in self.vocab: tokens.append(a_cuoi)

            if not tokens:
                final_tokens.append("[UNK]")
                final_ids.append(self.vocab["[UNK]"])
            else:
                final_tokens.extend(tokens)
                final_ids.extend([self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens])

        final_tokens.append("[SEP]")
        final_ids.append(self.vocab["[SEP]"])
        return final_ids

    def decode(self, ids: list) -> str:
        """Chuyển list id về câu hoàn chỉnh."""
        special = set(self.special_tokens.values())
        tokens  = [self.id_to_vocab.get(i, "") for i in ids if i not in special]
        return merge_subwords(tokens)
