southern-vietnamese-translator
---
## Architecture:
```plaintext 
southern-vietnamese-translator/
├── data/                       # Đổi tên cho ngắn gọn
│   ├── raw/                    # Chứa CSV thô ban đầu (Minh làm)
│   └── processed/              # Chứa CSV đã làm sạch & chia Train/Val/Test
│
├── notebooks/                  # Gom "khu vui chơi" của Jupyter vào 1 chỗ
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_error_analysis.ipynb
│
├── src/                        # TRÁI TIM DỰ ÁN: Gom toàn bộ code vào đây
│   ├── data/
│   │   ├── data_loader.py      # Script của Mai
│   │   └── tokenize_vocab.py   # Script của Ngọc
│   ├── models/                 # Chỉ chứa Deep Learning
│   │   ├── seq2seq_lstm.py
│   │   ├── transformer_full.py
│   │   └── transformer_lora.py
│   ├── baselines/              
│   │   └── rule_based.py       # Tách riêng thuật toán If-Else/Từ điển ra
│   └── utils.py                # Hàm hỗ trợ chung
│
├── outputs/                    # (MỚI) Nơi chứa TẤT CẢ sản phẩm sinh ra
│   ├── checkpoints/            # Nơi lưu file weights (.pt, .bin) khi train
│   ├── logs/                   # Nơi lưu Tensorboard/Loss để vẽ biểu đồ
│   └── predictions/            # Lưu file dịch của tập Test để chạy đánh giá
│
├── config.py                   # Bảng điều khiển trung tâm (để ngoài cùng cho dễ sửa)
├── train.py                    # Script kích hoạt huấn luyện (Gọi code từ src/)
├── evaluate.py                 # Script kích hoạt đánh giá (Gọi code từ src/)
├── app.py                      # Dùng tao wed Gradio trên Hugging Face Spaces
├── requirements.txt
└── README.md
```
