import re
import random

class RuleBasedTranslator:
    def __init__(self, dictionary_df):
        """
        dictionary_df: Dataframe chứa 2 cột 'Từ' và 'Ý nghĩa'
        """
        self.translation_map = {}
        self._build_mapping(dictionary_df)

    def _build_mapping(self, df):
        """Xây dựng bảng tra cứu tối ưu"""
        # Loại bỏ dòng bị khuyết nếu có (Nhớ đổi tên cột cho đúng data của bà)
        clean_df = df.dropna(subset=["word", "meaning"])

        for _, row in clean_df.iterrows():
            key = str(row["word"]).lower().strip()
            val = str(row["meaning"]).lower().strip()

            # --- XỬ LÝ TẠO ARRAY Ở ĐÂY ---
            # Dùng Regex để cắt chuỗi.
            # r',|\b\d+[\.\)]\s*' nghĩa là: cắt khi gặp dấu phẩy (,) HOẶC số kèm dấu chấm (1. 2.)
            raw_meanings = re.split(r',|\b\d+[\.\)]\s*', val)

            # Thêm .replace(".", "") để xóa sạch dấu chấm
            # và .strip(" ;") để gọt bỏ khoảng trắng, dấu chấm phẩy thừa ở 2 đầu
            meanings_list = [m.replace(".", "").strip(" ;") for m in raw_meanings if m.replace(".", "").strip(" ;")]

            # Đề phòng trường hợp cắt xong mảng bị rỗng, ta gán lại mảng chứa chuỗi gốc
            if not meanings_list:
                meanings_list = [key]

            # Lưu vào dictionary dưới dạng: { "từ": ["nghĩa 1", "nghĩa 2"] }
            self.translation_map[key] = meanings_list

        # Sắp xếp các từ khóa theo độ dài giảm dần (từ nhiều chữ nhất lên đầu)
        self.sorted_keys = sorted(self.translation_map.keys(), key=len, reverse=True)

    def translate_sentence(self, text):
        """Dịch một câu văn bản dựa trên phương pháp duyệt và thay thế chuỗi"""
        if not text:
            return ""

        # Chuẩn hóa văn bản input dạng chữ thường
        translated_text = text.lower().strip()

        # Duyệt qua từng từ khóa trong từ điển đã sắp xếp
        for western_word in self.sorted_keys:
            # Lấy ra mảng các ý nghĩa của từ này
            meanings_list = self.translation_map[western_word]

            # --- BỐC NGẪU NHIÊN 1 NGHĨA TRONG ARRAY ---
            vietnamese_common = random.choice(meanings_list)

            # Sử dụng Regex lookbehind và lookahead
            pattern = r'(?<![\wàáảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])' \
                      + re.escape(western_word) \
                      + r'(?![\wàáảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])'

            translated_text = re.sub(pattern, vietnamese_common, translated_text)

        # Bỏ khoảng trắng thừa nếu có
        translated_text = re.sub(r'\s+', ' ', translated_text).strip()
        return translated_text

    def evaluate_batch(self, test_dataframe):
        """
        Hàm chạy thử trên tập test để sinh kết quả dịch Baseline
        """
        predictions = []
        for _, row in test_dataframe.iterrows():
            # Bà nhớ check lại tên cột trong tập test của bà xem là "Input" hay tên gì nha
            source_text = str(row["Input"])
            pred = self.translate_sentence(source_text)
            predictions.append(pred)

        return predictions
