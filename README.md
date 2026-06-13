# Hệ thống Dịch máy Phương ngữ Miền Tây sang Tiếng Việt Phổ thông

Dự án nghiên cứu và xây dựng hệ thống Dịch máy Nơ-ron (NMT) chuyên biệt nhằm chuyển đổi các câu thoại mang đậm đặc trưng phương ngữ Nam Bộ (Đồng bằng sông Cửu Long) về ngôn ngữ tiếng Việt chuẩn phổ thông. Hệ thống thực hiện đánh giá, đối chứng hiệu năng toàn diện qua 4 cấp độ tiếp cận công nghệ có mức độ phức tạp tăng dần.

---

## 📌 Các tính năng & Mô hình triển khai đối chứng
Hệ thống triển khai thực nghiệm song song 4 phương pháp cốt lõi:
1.  **Rule-Based Baseline:** Sử dụng tập luật ánh xạ từ vựng cố định (Dictionary Mapping) dựa trên ngữ liệu chuyên khảo để thiết lập đường cơ sở.
2.  **Seq2Seq LSTM (Huấn luyện từ đầu):** Kiến trúc mạng hồi quy tuần tự với **Bi-LSTM Encoder** (đọc ngữ cảnh lập thể hai chiều) kết hợp cơ chế tập trung toàn cục **Luong Attention**, tầng cầu nối phi tuyến (Bridge Layer), và kỹ thuật **Input Feeding** nâng cao tại tầng đầu ra.
3.  **Transformer Full Fine-Tuning (viT5-Base):** Áp dụng học chuyển giao (Transfer Learning), tinh chỉnh toàn bộ 226M tham số của kiến trúc Transformer được tiền huấn luyện tối ưu cho tiếng Việt.
4.  **Transformer LoRA (Low-Rank Adaptation):** Sử dụng kỹ thuật tinh chỉnh hiệu quả tham số (PEFT), đóng băng ma trận trọng số gốc $W_0$ và chỉ cập nhật nhánh song song tích hạng thấp $BA$ trên các ma trận chiếu $\{W_Q, W_V\}$ nhằm tối ưu hóa bộ nhớ GPU VRAM.

---

## 📊 Pipeline và Chiến lược Dữ liệu
Để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (*Data Leakage*), quy trình được thiết kế cô lập nghiêm ngặt theo các giai đoạn:
* **Ngữ liệu gốc ban đầu:** Gồm **10.511 cặp câu** song ngữ được trích xuất từ cuốn *"Từ điển từ ngữ Nam Bộ"* của PGS.TS Huỳnh Công Tín, xử lý dịch chuyển ngữ và tinh lọc tự động thông qua LLM.
* **Phân chia dữ liệu (Data Splitting):** Thực hiện phân rã tập dữ liệu theo tỷ lệ **80% Train / 10% Validation / 10% Testing** trực tiếp trên tập gốc sạch trước khi tăng cường dữ liệu. Tập Validation và Test được đóng băng nguyên bản 100%.
* **Xác lập ngưỡng chặn chuỗi:** Dựa vào phân tích đồ thị mật độ tích lũy phân vị $P95$, tham số `Max_Sequence_Length` được cấu hình cố định ở mức **17 từ** dựa trên phân vị $P95$ của câu ví dụ phương ngữ nhằm tối ưu dung lượng GPU VRAM.
* **Tăng cường dữ liệu (Data Augmentation):** Chỉ áp dụng kỹ thuật sinh câu biến thể bằng LLM độc lập trên tập Train gốc nhằm mở rộng quy mô từ tập huấn luyện ban đầu lên thành **25.821 mẫu**, giúp nâng cao khả năng tổng quát hóa ngôn ngữ.

---

## 🛠️ Cấu hình Kỹ thuật và Tối ưu hóa
Hệ thống huấn luyện trên môi trường phần cứng **GPU Tesla T4 (16GB VRAM)** tích hợp các giải pháp lập trình nâng cao:
* **Bộ tối ưu AdamW:** Tách biệt phân rã trọng số (`weight_decay = 1e-4` cho các ma trận cốt lõi) và loại trừ phân rã trọng số (`weight_decay = 0.0` cho tập nhiễu cấu trúc như `bias`, `LayerNorm.weight`, `layer_norm.weight`).
* **Điều phối tốc độ học (LR Scheduler):** Triển khai chu kỳ `LambdaLR` với **Linear Warmup** trong 10% bước đầu để tránh bùng nổ gradient, kết hợp **Linear Decay/Cosine Annealing** để hạ tốc độ học mịn màng về 0.
* **Chiến lược Scheduled Sampling (Cho Seq2Seq):** Cài đặt hàm giảm tuyến tính hệ số `Teacher Forcing ratio` (`get_tf_ratio`) theo từng epoch để khắc phục triệt để lỗi lệch pha phân phối giữa huấn luyện và suy luận (*Exposure Bias*).
* **Tiết kiệm bộ nhớ đồ họa:** Tích hợp cơ chế tính toán độ chính xác hỗn hợp **Mixed Precision (FP16/BF16)** và **Gradient Checkpointing** để hoán đổi bộ nhớ lấy tài nguyên tính toán trong pha backward pass của Transformer.

---
## 📁 Cấu trúc thư mục Dự án

```text
├── data/
│   ├── processed/
│   │   ├── test.csv                    # Tập dữ liệu kiểm thử độc lập (10% gốc)
│   │   ├── train.csv                   # Tập dữ liệu huấn luyện sau khi tăng cường (25k)
│   │   ├── val.csv                     # Tập dữ liệu kiểm chứng độc lập (10% gốc)
│   │   └── vocab_word_level.json       # Từ điển cấp độ từ vựng được khởi tạo
│   └── data_vietnamese_dialect.csv     # Tập dữ liệu song ngữ gốc ban đầu (10.511 mẫu)
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb       # Khám phá dữ liệu (EDA) và tiền xử lý văn bản
│   ├── 02_error_analysis.ipynb         # Notebook hỗ trợ phân tích định tính kết quả sai
│   ├── [DL]_EDA__Pre.ipynb             # Notebook nháp/thử nghiệm EDA ban đầu
│   └── error_analysis.ipynb            # Script phân tích và trực quan hóa lỗi dịch thuật
│
├── outputs/checkpoints/
│   └── logs/
│       └── predictions/                # Log lưu trữ các chuỗi ký tự mô hình dự đoán
│           └── placeholder
│
└── source/
    ├── baselines/
    │   └── rule_based.py               # Phương pháp dịch dựa trên luật (Baseline)
    ├── data/
    │   ├── data_loader.py              # Xử lý nạp dữ liệu, tạo Batch và xử lý Padding/Masking
    │   └── tokenize_vocab.py           # Bộ tách từ và quản lý không gian từ vựng
    ├── models/
    │   ├── seq2seq_lstm.py             # Kiến trúc mạng Seq2Seq Bi-LSTM + Luong Attention
    │   ├── transformer_full.py         # Kiến trúc tinh chỉnh toàn bộ tham số (Full Fine-Tuning)
    │   └── transformer_lora.py         # Kiến trúc tinh chỉnh tham số hiệu quả qua LoRA (PEFT)
    ├── .gitignore                      # Cấu hình bỏ qua các tệp tin rác khi push GitHub
    ├── LICENSE                         # Giấy phép bản quyền của dự án
    ├── README.md                       # Tài liệu hướng dẫn dự án (File này)
    ├── app.py                          # Ứng dụng demo chạy thực tế
    ├── config.py                       # Quản lý tập trung hệ thống siêu tham số (Hyperparameters)
    ├── evaluate.py                     # Script tính toán các thang đo định lượng (BLEU, ROUGE-L)
    ├── plot_loss.py                    # Script vẽ biểu đồ đường cong hội tụ (Learning Curves)
    └── utils.py                        # Chứa các hàm bổ trợ (Optimizer, Scheduler, get_tf_ratio)
```
🚀 Hướng dẫn Cài đặt & Sử dụng
1. Cài đặt môi trường
Yêu cầu hệ thống cài đặt sẵn Python >= 3.8 và môi trường tính toán CUDA (Khuyến nghị sử dụng GPU Tesla T4 hoặc tương đương).
Bash pip install -r requirements.txt

2. Khám phá và Tiền xử lý Dữ liệu
Mở và chạy file Notebook để theo dõi biểu đồ phân phối EDA cấp độ ký tự, số từ và phân tán tương quan độ dài câu:
Bash jupyter notebook notebooks/01_eda_and_cleaning.ipynb

3. Huấn luyện mô hình
Chạy cấu trình huấn luyện mô hình 
Bash python train.py

4. Thử nghiệm dịch câu (Suy luận)
Sử dụng script translate.py để dịch một câu phương ngữ Miền Tây ngẫu nhiên bằng thuật toán Beam Search (kích thước beam_size = 4):
Bash python translate.py --text "tao đi mần nha" --method beam
