import torch 
import torch.nn as nn
import torch.nn.functional as F 
import random

#---------------------ENCODER---------------------#
class Encoder(nn.Module):
    """
    Bi-LSTM Encoder — đọc chuỗi input và nén thành ngữ cảnh.

    Input : "đặng [SEP] có thể được [SEP] tao đặng làm việc đó"
             (đã qua tokenize)

    Output:
      - encoder_outputs : (B, src_len, hidden*2)  : dùng cho Attention
      - hidden: (B, hidden) : khởi tạo Decoder
      - cell: (B, hidden): khởi tạo Decoder

    Dùng Bi-LSTM để hiểu ngữ cảnh 2 chiều
        Đọc cả 2 chiều: "đặng" hiểu nghĩa tốt hơn khi biết cả từ trước (từ miền Tây) lẫn từ sau (giải thích sau [SEP])
    """
    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int,
        hidden_dim : int,
        n_layers   : int,
        dropout    : float,
        pad_idx    : int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Embedding: token ID --> vector liên tục
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx)
        '''
        vocab_size: số lượng từ trong từ điển, mỗi từ có 1 hàng trong ma trận embedding
        embed_dim: số chiều của vector biểu diễn của 1 từ
        '''

        # Bi-LSTM: mỗi chiều có hidden_dim --> output = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,       # (B, L, H) thay vì (L, B, H)
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        '''
        batch_first: thứ tự chiều của Tensor (Batch, Length, Hidden)
        '''

        # Bridge: thu gọn Bi-LSTM hidden (hidden*2) về hidden_dim để Decoder (LSTM 1 chiều) nhận được
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, hidden_dim) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple:
        """
        src: (B, src_len) — batch token IDs, src_len là độ dài của từ dài nhất trong batch

        Returns:
          encoder_outputs : (B, src_len, hidden*2)
          hidden          : (n_layers, B, hidden)
          cell            : (n_layers, B, hidden)
        """
        # (B, src_len) → (B, src_len, embed_dim)
        embedded = self.dropout(self.embedding(src)) # dùng dropout để tăng khả năng tổng quát hóa

        # LSTM chạy qua toàn bộ chuỗi
        # outputs: (B, src_len, hidden*2), để cho Attention
        # h_n, c_n: (n_layers*2, B, hidden), truyền cho decoder
        outputs, (h_n, c_n) = self.lstm(embedded)

        # ── Xử lý hidden để đưa vào Decoder ──────────────────────
        # h_n có shape: (n_layers*2, B, hidden)
        # Tách forward và backward của mỗi layer rồi ghép lại
        # Layer i forward  = h_n[2*i]
        # Layer i backward = h_n[2*i + 1]

        hidden_list = []
        cell_list   = []
        for i in range(self.n_layers):
            # Ghép forward + backward → (B, hidden*2)
            h_forward  = h_n[2 * i]        # (B, hidden)
            h_backward = h_n[2 * i + 1]    # (B, hidden)
            h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (B, hidden*2)

            c_forward  = c_n[2 * i]
            c_backward = c_n[2 * i + 1]
            c_cat = torch.cat([c_forward, c_backward], dim=-1)

            hidden_list.append(self.tanh(self.fc_hidden(h_cat)))
            cell_list.append(self.tanh(self.fc_cell(c_cat)))

        # Stack lại: (n_layers, B, hidden)
        hidden = torch.stack(hidden_list, dim=0)
        cell   = torch.stack(cell_list,   dim=0)

        return outputs, hidden, cell


#---------------------ATTENTION ---------------------#
class LuongAttention(nn.Module):
    '''
    Công thức: Luong Attention
    e_t,i=(s_t)T*(W_a*h_i) 
    alpha_t,i=softmax(e_t,i): chuyển sang hàm softmax để tính trọng số
    c_t=sum(alpha_t,i*h_i): công thức tính context là tổ hợp tuyến tính của trọng số với trạng thái ấn của encoder
    '''
    def __init__(self, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, hidden_dim, bias=False)

    def forward(
        self,
        decoder_hidden  : torch.Tensor, # (B, hidden_dim)
        encoder_outputs : torch.Tensor,  # (B, src_len, encoder_dim)
        src_mask        : torch.Tensor   ) -> tuple:# (B, src_len) #tạm thời
        # (B, src_len, encoder_dim) --> (B, src_len, hidden_dim)
            h = self.W_a(encoder_outputs) # W_a*h_i --> đưa encoder_dim và hidden_dim
            s = decoder_hidden.unsqueeze(1)  #(B, 1, hidden_dim)

        # (B, src_len, hidden_dim) --> (B, hidden_dim, src_len)
            h_transposed = h.transpose(1, 2)

        # (B, 1, hidden) x (B, hidden, src_len) = (B, 1, src_len) - e_t,i=(s_t)T*W_a*h_i
            energy = torch.bmm(s, h_transposed).squeeze(1)

        # Mask: đặt score của PAD token thành - inf trước softmax
        # → attention weight của PAD = 0 sau softmax
            energy = energy.masked_fill(src_mask == 0, float('-inf')) # tạm thời sử dụng cái này

            attn_weights = F.softmax(energy, dim=-1) #đưa vào hàm softmax để tính trong số

        # (B, 1, src_len) x (B, src_len, encoder_dim) --> (B, encoder_dim) - c_t=sum(alpha_t,i*h_i)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            return context, attn_weights # trả về vector ngữ cảnh và trọng số
    
#----------------------DECODER---------------------
class Decoder(nn.Module):
    """
    LSTM Decoder với Attention — sinh từng token của bản dịch.
    Ở mỗi bước t:
      1. Embedding của token trước: y_{t-1} --> emb
      2. Tính Attention với encoder outputs --> context c_t
      3. LSTM nhận [emb ; c_t] --> hidden s_t
      4. Linear(s_t) --> logits over vocab

    Tại sao concat [emb; c_t] thay vì chỉ emb?
      Decoder biết phải "dịch phần nào" của input ở mỗi bước
      Không bị "quên" thông tin input sau nhiều bước decode
    """
    def __init__(
        self,
        vocab_size  : int,
        embed_dim   : int,
        hidden_dim  : int,
        encoder_dim : int,    # hidden*2 từ Encoder
        n_layers    : int,
        dropout     : float,
        pad_idx     : int):
        
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.attention = LuongAttention(hidden_dim, encoder_dim)
        # Input = emb + context → embed_dim + encoder_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        # Output layer: hidden → vocab logits
        # Dùng thêm emb và context để tăng thông tin (input feeding)
        self.fc_out = nn.Linear(
            hidden_dim + encoder_dim + embed_dim,
            vocab_size
        )
        """
        hidden_dim: s_t (từ LSTM)
        encoder_dim: h_i (từ encoder)
        embed_dim: emb (từ embedding y_t-1)
        """
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        input_token     : torch.Tensor,   # (B,) — 1 token
        hidden          : torch.Tensor,   # (n_layers, B, hidden)
        cell            : torch.Tensor,   # (n_layers, B, hidden)
        encoder_outputs : torch.Tensor,   # (B, src_len, encoder_dim)
        src_mask        : torch.Tensor    # (B, src_len)
    ) -> tuple:
        """
        Decode 1 bước → trả về logits + hidden mới + attention weights

        Returns:
          logits       : (B, vocab_size)
          hidden       : (n_layers, B, hidden)
          cell         : (n_layers, B, hidden)
          attn_weights : (B, src_len) ← dùng để visualize sau
        """
        # (B,) → (B, 1) → (B, 1, embed_dim)
        embedded = self.dropout(
            self.embedding(input_token.unsqueeze(1))
        )

        # Lấy hidden của layer trên cùng để tính Attention
        # hidden: (n_layers, B, hidden) → top layer: (B, hidden)
        top_hidden = hidden[-1]

        # Tính context vector từ Attention
        context, attn_weights = self.attention(
            top_hidden, encoder_outputs, src_mask
        )    # context: (B, encoder_dim)

        # Ghép embedding + context → LSTM input
        # (B, 1, embed_dim) + (B, 1, encoder_dim) → (B, 1, embed+enc)
        context_expanded = context.unsqueeze(1)   # (B, 1, encoder_dim)
        lstm_input = torch.cat([embedded, context_expanded], dim=-1)

        # LSTM forward 1 bước
        # output: (B, 1, hidden)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)    # (B, hidden)

        # Output layer: kết hợp hidden + context + embedding
        embedded_sq = embedded.squeeze(1)   # (B, embed_dim)
        logits = self.fc_out(
            torch.cat([output, context, embedded_sq], dim=-1)
        )    # (B, vocab_size)

        return logits, hidden, cell, attn_weights

#----------------------SEQUENCE-TO-SEQUENCE MODEL---------------------
# cái này tạm thời có mask và greedy để mô phỏng chạy thử, khi ghép module lại sẽ xóa, sửa 
class Seq2Seq(nn.Module):
    """
    Wrapper kết hợp Encoder + Decoder, quản lý Teacher Forcing.

    TEACHER FORCING:
      Training  : với xác suất tf_ratio, dùng ground truth làm
                  input bước tiếp theo thay vì dùng predicted token
                  → mô hình hội tụ nhanh hơn, ổn định hơn
      Inference : luôn dùng predicted token (tf_ratio = 0.0)
                  → Greedy hoặc Beam Search

    tf_ratio thường giảm dần theo epoch (Scheduled Sampling): --> để mô hình thích nghi với việc tự học sau khi được dạy
      Epoch 1-3 : 0.9  (học nhiều từ ground truth)
      Epoch 4-6 : 0.7
      Epoch 7+  : 0.5  (tự lực là chính)
    --> cái tf_ratio giảm dần này để set ở trong train.py để mô hình hội tụ nhanh hơn
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def make_src_mask(self, src: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """
        Tạo mask cho Attention: 1 = token thực, 0 = PAD
        src: (B, src_len)
        Returns: (B, src_len)
        """
        return (src != pad_idx)

    def forward(
        self,
        src       : torch.Tensor,    # (B, src_len)
        trg       : torch.Tensor,    # (B, trg_len)
        pad_idx   : int,             # tạm thời, sẽ thay thế
        tf_ratio  : float = 0.5      # Teacher Forcing ratio, phòng hờ khi quên truyền tham số
    ) -> torch.Tensor:
        """
        Returns:
          outputs : (B, trg_len-1, vocab_size)
                    → so sánh với trg[:, 1:] để tính loss
        """
        B, trg_len = trg.shape
        vocab_size  = self.decoder.vocab_size

        # Tensor chứa logits ở mỗi bước decode
        outputs = torch.zeros(B, trg_len - 1, vocab_size).to(self.device)

        # ── ENCODER PASS ────────────────────────────────────────
        encoder_outputs, hidden, cell = self.encoder(src)
        # encoder_outputs: (B, src_len, hidden*2)
        # hidden, cell   : (n_layers, B, hidden)

        # Mask cho Attention (che PAD tokens), TẠM
        src_mask = self.make_src_mask(src, pad_idx)    # (B, src_len)

        # ── DECODER PASS (từng bước) ────────────────────────────
        # Token đầu tiên luôn là <BOS>
        dec_input = trg[:, 0]    # (B,)  — cột 0 = <BOS>

        for t in range(trg_len - 1):
            # Decode 1 bước
            logits, hidden, cell, _ = self.decoder.forward_step(
                dec_input, hidden, cell, encoder_outputs, src_mask
            )
            # logits: (B, vocab_size)

            outputs[:, t, :] = logits

            # ── TEACHER FORCING DECISION ───────────────────────
            use_teacher = random.random() < tf_ratio

            if use_teacher:
                # Dùng ground truth token tiếp theo
                dec_input = trg[:, t + 1]       # (B,), ý là ngầm hiểu đưa từ đầu tiên vào làm gốc
            else:
                # Dùng token có xác suất cao nhất (Greedy)
                dec_input = logits.argmax(dim=-1)   # (B,) # những từ tiếp theo dùng beam/greedy để sinh từ

        return outputs    # (B, trg_len-1, vocab_size)

    @torch.no_grad() #TẠM
    def translate_greedy(
        self,
        src        : torch.Tensor,    # (1, src_len) — 1 câu
        pad_idx    : int,
        bos_idx    : int,
        eos_idx    : int,
        max_len    : int = 50
    ) -> list[int]:
        """
        Greedy Decoding — sinh bản dịch từng từ một.
        Dùng để demo và baseline eval.
        """
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src)
        src_mask = self.make_src_mask(src, pad_idx)

        dec_input = torch.tensor([bos_idx], device=self.device)
        translated_ids = []

        for _ in range(max_len):
            logits, hidden, cell, attn = self.decoder.forward_step(
                dec_input, hidden, cell, encoder_outputs, src_mask
            )
            pred_token = logits.argmax(dim=-1)    # (1,)
            token_id   = pred_token.item()

            if token_id == eos_idx:
                break

            translated_ids.append(token_id)
            dec_input = pred_token

        return translated_ids

