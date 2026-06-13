Ngon lành m ơi! T ráp toàn bộ các mảnh ghép từ nãy đến giờ lại thành một file `README.md` **full option, hoàn chỉnh 100%** không thiếu một chữ nào. Cấu trúc này ôm trọn bộ data 10.511 câu, ảnh EDA mới, con Seq2Seq **36M tham số**, thuật toán tối ưu, hướng dẫn chạy lệnh rạch ròi và cả phần phân tích 4 nhóm lỗi luôn nha.

M copy nguyên si toàn bộ nội dung trong khối code dưới đây tạo thành file `README.md` quăng thẳng vào thư mục gốc của dự án trên GitHub là chốt sổ điểm 10 luôn nè:

```markdown
# Hệ thống Dịch máy Phương ngữ Miền Tây sang Tiếng Việt Phổ thông: Thử nghiệm đối chứng đa kiến trúc (Comprehensive Benchmark)

Dự án nghiên cứu và xây dựng hệ thống Dịch máy Nơ-ron (NMT) chuyên biệt nhằm chuyển đổi các câu thoại mang đậm đặc trưng phương ngữ Nam Bộ (Đồng bằng sông Cửu Long) về ngôn ngữ tiếng Việt chuẩn phổ thông. Hệ thống thực hiện đánh giá, đối chứng hiệu năng toàn diện qua 4 cấp độ tiếp cận công nghệ có mức độ phức tạp tăng dần.

---

## 📌 Các tính năng & Mô hình triển khai đối chứng
Hệ thống triển khai thực nghiệm song song 4 phương pháp cốt lõi:
1.  **Rule-Based Baseline:** Sử dụng tập luật ánh xạ từ vựng cố định (Dictionary Mapping) dựa trên ngữ liệu chuyên khảo để thiết lập đường cơ sở.
2.  **Seq2Seq LSTM (Huấn luyện từ đầu - 36M Parameters):** Kiến trúc mạng hồi quy tuần tự cấu hình lớn với **Bi-LSTM Encoder** (đọc ngữ cảnh lập thể hai chiều), tầng cầu nối phi tuyến (**Bridge Layer**), cơ chế tập trung toàn cục **Luong Attention** tích hợp Masking hệ số $-\infty$, và kỹ thuật **Input Feeding** nâng cao tại tầng đầu ra.
3.  **Transformer Full Fine-Tuning (viT5-Base - 226M Parameters):** Áp dụng học chuyển giao (Transfer Learning), tinh chỉnh toàn bộ 226M tham số của kiến trúc Transformer được tiền huấn luyện tối ưu cho tiếng Việt.
4.  **Transformer LoRA (Low-Rank Adaptation - PEFT):** Sử dụng kỹ thuật tinh chỉnh hiệu quả tham số, đóng băng ma trận trọng số gốc $W_0$ và chỉ cập nhật nhánh song song tích hạng thấp $BA$ trên các ma trận chiếu $\{W_Q, W_V\}$ nhằm tối ưu hóa bộ nhớ GPU VRAM.

---

## 📊 Pipeline và Chiến lược Dữ liệu
Để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (*Data Leakage*), quy trình được thiết kế cô lập nghiêm ngặt theo các giai đoạn:
* **Ngữ liệu gốc ban đầu:** Gồm **10.511 cặp câu** song ngữ được trích xuất từ cuốn *"Từ điển từ ngữ Nam Bộ"* của PGS.TS Huỳnh Công Tín, xử lý dịch chuyển ngữ và tinh lọc tự động thông qua LLM.
* **Phân chia dữ liệu (Data Splitting):** Thực hiện phân rã tập dữ liệu theo tỷ lệ nghiêm ngặt **80% Train / 10% Validation / 10% Testing** trực tiếp trên tập gốc sạch trước khi tăng cường dữ liệu. Tập Validation và Test được đóng băng nguyên bản 100%.
* **Xác lập ngưỡng chặn chuỗi:** Dựa vào phân tích đồ thị mật độ tích lũy phân vị $P95$ của câu ví dụ phương ngữ, tham số `Max_Sequence_Length` được cấu hình cố định ở mức **17 từ** nhằm tối ưu dung lượng GPU VRAM và triệt tiêu các mã đệm `<PAD>` vô nghĩa.
* **Tăng cường dữ liệu (Data Augmentation):** Chỉ áp dụng kỹ thuật sinh câu biến thể bằng LLM độc lập trên tập Train gốc nhằm mở rộng quy mô từ tập huấn luyện ban đầu lên thành **25.821 mẫu**, giúp nâng cao khả năng tổng quát hóa ngôn ngữ.

---

## 🛠️ Cấu hình Kỹ thuật và Tối ưu hóa
Hệ thống huấn luyện trên môi trường phần cứng **GPU Tesla T4 (16GB VRAM)** tích hợp các giải pháp lập trình nâng cao trong file `utils.py`:
* **Bộ tối ưu AdamW:** Tách biệt phân rã trọng số (`weight_decay = 1e-4` cho các ma trận cốt lõi) và loại trừ phân rã trọng số (`weight_decay = 0.0` cho tập nhiễu cấu trúc như `bias`, `LayerNorm.weight`, `layer_norm.weight`).
* **Điều phối tốc độ học (LR Scheduler):** Triển khai chu kỳ `LambdaLR` với **Linear Warmup** trong 10% bước đầu để tránh bùng nổ gradient khi trọng số chưa ổn định, kết hợp **Linear Decay** để hạ tốc độ học mịn màng về 0.
* **Chiến lược Scheduled Sampling (Cho Seq2Seq):** Cài đặt hàm giảm tuyến tính hệ số `Teacher Forcing ratio` (`get_tf_ratio`) theo từng epoch nhằm giải quyết hiện tượng lệch pha phân phối giữa pha huấn luyện và pha suy luận (*Exposure Bias*).
* **Tiết kiệm bộ nhớ đồ họa:** Tích hợp cơ chế tính toán độ chính xác hỗn hợp **Mixed Precision (FP16/BF16)** và **Gradient Checkpointing** để hoán đổi bộ nhớ lấy tài nguyên tính toán trong pha backward pass của Transformer.

---

## 📁 Cấu trúc thư mục Dự án

```text
├── data/
│   ├── processed/
│   │   ├── test.csv                    # Tập dữ liệu kiểm thử độc lập (10% gốc)
│   │   ├── train.csv                   # Tập dữ liệu huấn luyện sau khi tăng cường (25.821 mẫu)
│   │   ├── val.csv                     # Tập dữ liệu kiểm chứng độc lập (10% gốc)
│   │   └── vocab_word_level.json       # Từ điển cấp độ từ vựng được khởi tạo cho Seq2Seq
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
    │   ├── seq2seq_lstm.py             # Kiến trúc mạng Seq2Seq Bi-LSTM + Luong Attention (36M)
    │   ├── transformer_full.py         # Kiến trúc tinh chỉnh toàn bộ tham số (Full Fine-Tuning)
    │   └── transformer_lora.py         # Kiến trúc tinh chỉnh tham số hiệu quả qua LoRA (PEFT)
    ├── .gitignore                      # Cấu hình bỏ qua các tệp tin rác khi push GitHub
    ├── LICENSE                         # Giấy phép bản quyền của dự án
    ├── README.md                       # Tài liệu hướng dẫn dự án (File này)
    ├── app.py                          # Ứng dụng giao tiếp hoặc demo chạy thực tế
    ├── config.py                       # Quản lý tập trung hệ thống siêu tham số (Hyperparameters)
    ├── evaluate.py                     # Script tính toán các thang đo định lượng (BLEU, ROUGE-L)
    ├── plot_loss.py                    # Script vẽ biểu đồ đường cong hội tụ (Learning Curves)
    └── utils.py                        # Chứa các hàm bổ trợ (Optimizer, Scheduler, get_tf_ratio)

```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường

Yêu cầu hệ thống cài đặt sẵn Python >= 3.8 và môi trường tính toán CUDA (Khuyến nghị sử dụng GPU Tesla T4 hoặc tương đương để chạy mô hình Transformer/LoRA và kiến trúc Seq2Seq lớn).

```bash
# Khởi tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

```

### 2. Khám phá và Tiền xử lý Dữ liệu

Mở và chạy file Notebook để theo dõi biểu đồ phân phối EDA (Phân phối độ dài từ, số từ câu nguồn/đích và biểu đồ phân tán tương quan độ dài câu):

```bash
jupyter notebook notebooks/01_eda_and_cleaning.ipynb

```

### 3. Huấn luyện mô hình (Training) 
Vào config đổi CHOSEN thành model muốn chạy
Sau đó
```bash
py train.py

```


### 4. Thử nghiệm dịch câu (Suy luận / Inference)

#### Sử dụng mô hình Seq2Seq LSTM (36M Tham số):

* **Chế độ giải mã tham lam (Greedy Search):**
```bash
python source/translate.py --model seq2seq --text "hồi sớm mai tao thấy nó đặng làm việc đó" --method greedy

```


* **Chế độ giải mã chùm (Beam Search - Kích thước beam = 4, tích hợp Length Penalty):**
```bash
python source/translate.py --model seq2seq --text "hồi sớm mai tao thấy nó đặng làm việc đó" --method beam --beam_size 4

```



#### Sử dụng mô hình Transformer LoRA:

Hệ thống sẽ tự động thực hiện bước hợp nhất ma trận trọng số ($\text{W}_{\text{deploy}} = \text{W}_0 + \frac{\alpha}{r}\text{BA}$) để suy diễn độc lập với tốc độ tối ưu, không phát sinh thêm độ trễ (latency):

```bash
python source/translate.py --model lora --text "ngày nào cũng lo ăn cho bấy nhiêu người chạy vầy là khẳm thiệt chớ"

```

---

## 📈 Kết quả Thực nghiệm Định lượng

Hiệu suất kiểm thử thực tế trên tập dữ liệu thử nghiệm (Test Set) độc lập:

| Mô hình / Phương pháp | Số lượng tham số | Thời gian huấn luyện | BLEU (%) | ROUGE-L (%) | Đặc trưng hội tụ (Learning Curves) |
| --- | --- | --- | --- | --- | --- |
| **Rule-Based Baseline** | -- | -- | 12.45% | 34.20% | Không có pha học tập, tra cứu cứng nhắc. |
| **Seq2Seq LSTM** | **~36M** | ~45 phút | 26.80% | 50.15% | Giảm loss chậm, kẹt loss ở mức cao (4.x), overfit sớm do không gian tham số quá lớn so với tập data thưa thớt. |
| **Transformer Full (ViT5)** | ~226M | ~2.5 giờ | 37.15% | 62.40% | Giảm loss nhanh ban đầu nhưng bất ổn định, kích hoạt Early Stopping. |
| **Transformer LoRA (ViT5)** | **~226M (Trained 0.39%)** | **~4.5 giờ** | **41.46%** | **68.20%** | **Tối ưu ổn định vượt trội, đường cong Train/Val Loss ôm sát nhau mịn màng.** |

---

## 🔍 Phân tích định tính & Phân loại lỗi (Error Analysis)

Dựa trên việc kiểm tra và gán nhãn thủ công 100 trường hợp dự đoán sai ngẫu nhiên trích xuất từ file `phan_loai_loi_dich_lora.csv`, nhóm xác định hiệu năng hệ thống bị ràng buộc bởi 4 nhóm lỗi bản chất ngôn ngữ:

1. **Lỗi OOV (Out of Vocabulary):** Lúng túng trước từ lóng, từ trại phát âm hiếm gặp (*mụt* $\rightarrow$ giữ nguyên thay vì dịch thành *mầm*; *nhau nháu* $\rightarrow$ đoán mò thành *nhau nhõng nhẽo*).
2. **Lỗi Hallucination (Sinh ảo):** Đặc tính của Generative AI cố gắng tự làm mượt văn bản dẫn đến suy diễn sai lệch ngữ cảnh (*khẳm* $\rightarrow$ dịch thành *mệt* làm mất nét nghĩa bở hơi tai/quá tải; *lui hửi* $\rightarrow$ biến thành *lượn lờ* thay vì sục sạo ngửi).
3. **Lỗi Polysemy (Đa nghĩa / Sai bối cảnh):** Chọn nhầm nét nghĩa khi từ địa phương chuyển dịch môi trường giao tiếp (*tráo* mang nghĩa xảo trá bị dịch nhầm thành *trố mắt*).
4. **Lỗi Literal Translation (Dịch sát nghĩa đen):** Dịch thô bạo từng từ một (Word-by-word) làm câu đầu ra tiếng Phổ thông bị sượng và mất đi tính biểu cảm tự nhiên.

---

## 🛠️ Các thư viện sử dụng chính

* `torch >= 2.0` (PyTorch framework cấu trúc mạng cốt lõi)
* `transformers`, `peft` (Quản lý, nạp tri thức nền viT5 và thiết lập cấu hình LoRA adapter)
* `pandas`, `numpy` (Xử lý cấu trúc bảng ma trận và dữ liệu)
* `matplotlib`, `seaborn` (Trực quan hóa đồ thị hội tụ loss và phân phối EDA)

```

