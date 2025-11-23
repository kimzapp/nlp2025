# Lab 6: Giới thiệu về Transformers - Báo cáo

## Bài 1: Khôi phục Masked Token (Masked Language Modeling)

### Kết quả thực hiện

Sau khi chạy đoạn code mẫu với pipeline `fill-mask`, kết quả thu được như sau:

**Câu gốc:** Hanoi is the <mask> of Vietnam.

**Top 5 dự đoán:**

| Dự đoán     | Độ tin cậy | Câu hoàn chỉnh                     |
|-------------|------------|------------------------------------|
| capital     | 0.9341     | Hanoi is the capital of Vietnam.   |
| Republic    | 0.0300     | Hanoi is the Republic of Vietnam.  |
| Capital     | 0.0105     | Hanoi is the Capital of Vietnam.   |
| birthplace  | 0.0054     | Hanoi is the birthplace of Vietnam.|
| heart       | 0.0014     | Hanoi is the heart of Vietnam.     |

### Trả lời câu hỏi

#### 1. Mô hình đã dự đoán đúng từ `capital` không?

**Có**, mô hình đã dự đoán chính xác từ `capital` với độ tin cậy rất cao (93.41%). Đây là từ đúng nhất và phù hợp nhất về mặt ngữ nghĩa và địa lý, vì Hà Nội chính là thủ đô của Việt Nam.

Các dự đoán khác như `Republic`, `Capital` (viết hoa), `birthplace` và `heart` tuy có thể chấp nhận được trong một số ngữ cảnh khác nhưng không chính xác bằng `capital` trong câu hỏi này.

#### 2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

Các mô hình Encoder-only (như BERT) đặc biệt phù hợp cho tác vụ Masked Language Modeling (MLM) vì những lý do sau:
*   **Mục tiêu huấn luyện (Pre-training Objective):** Các mô hình Encoder-only như BERT được huấn luyện chính trên nhiệm vụ Masked Language Model. Trong quá trình này, một tỷ lệ phần trăm các từ trong câu đầu vào bị ngẫu nhiên che đi, và nhiệm vụ của mô hình là dự đoán chính xác những từ đó dựa trên ngữ cảnh xung quanh. Vì vậy, kiến trúc và cơ chế của nó được tối ưu hóa tự nhiên cho tác vụ này.

*   **Kiến trúc Bidirectional (Hai chiều):** Khác với các mô hình Decoder chỉ nhìn về quá khứ (từ trái sang phải), Encoder của BERT có thể xem xét toàn bộ ngữ cảnh của câu - cả các từ đứng **trước** và **sau** token `[MASK]`. Trong ví dụ "Hanoi is the [MASK] of Vietnam", để dự đoán từ bị che, mô hình cần hiểu:
    *   "Hanoi" là một địa danh.
    *   "Vietnam" là một quốc gia.
    *   Mối quan hệ giữa một thành phố và một quốc gia thường là "thủ đô - thủ đô".
    Khả năng "nhìn thấy" cả cụm "of Vietnam" là yếu tố then chốt giúp BERT đưa ra dự đoán `capital` một cách chính xác.

*   **Sự hiểu biết sâu sắc về ngữ cảnh:** Cơ chế Self-Attention trong Encoder cho phép mô hình đánh giá mức độ quan trọng của tất cả các từ trong câu đối với vị trí đang được xử lý (token `[MASK]`). Điều này giúp tạo ra các biểu diễn từ giàu ngữ cảnh, làm cơ sở cho việc dự đoán chính xác.

---------------------------------------------------
## Bài 2. Dự đoán từ tiếp theo (Next Token Prediction)

### Kết quả thực hiện

Sau khi chạy đoạn code mẫu với pipeline `text-generation`, kết quả thu được như sau:

**Câu mồi:** The best thing about learning NLP is...

**Câu hoàn thiện:**

```
The best thing about learning NLP is that your brain will not automatically switch to it when you think of it, as you'll be distracted by it.

You never know when you'll be able to pick up on NLP. It may take a little bit of practice to figure out what's going to get you going, but that doesn't mean you have to go through your training.

Step 4:

Try a different approach for your problem, and then learn more about your situation

When you're trying to learn a new activity, or new skill, you're learning to learn a specific activity because you can't do it in a specific way. You can learn these things through the practice of NLP.

If you're new to learning NLP, it's important to have a better idea of where your NLP activity is going. It gets better as you learn more about your activity.

If you're doing a lot of reading and listening, or more importantly, you're learning about your mental state, and you want to learn about the mental state of your mental body, then you may want to try learning about the mental state of your brain.

There's no such thing as a "naked" brain. It's the state that is most
```
### Trả lời câu hỏi
#### 1. Kết quả sinh ra có hợp lý không?

**Không**, kết quả sinh ra không thực sự hợp lý và thiếu tính mạch lạc.

**Phân tích chi tiết:**

**Ưu điểm:**
- Mô hình đã tiếp tục được câu mồi một cách ngữ pháp đúng

**Nhược điểm nghiêm trọng:**
- **Nội dung không logic:** Câu "your brain will not automatically switch to it when you think of it" trái ngược với mong đợi về lợi ích của việc học NLP
- **Thiếu tập trung:** Văn bản lan man sang các chủ đề không liên quan như "try a different approach", "mental state", "naked brain"
- **Không trả lời được câu hỏi ngầm định:** Câu "The best thing about learning NLP is..." nên được tiếp nối bằng một lợi ích cụ thể
- **Lặp từ và ý tưởng:** Các ý tưởng được lặp lại một cách không cần thiết

**Nguyên nhân:**
- Mô hình GPT-2 mặc định được sử dụng có thể chưa đủ mạnh
- Thiếu fine-tuning trên dữ liệu chuyên ngành
- Cần điều chỉnh hyperparameters để có kết quả tốt hơn

#### 2. Tại sao mô hình Decoder-only như GPT phù hợp cho tác vụ này?

Các mô hình Decoder-only như GPT đặc biệt phù hợp cho tác vụ sinh văn bản (Text Generation) vì những lý do sau:

**a. Tính tự hồi quy (Autoregressive Nature):**
- GPT được thiết kế để dự đoán token tiếp theo dựa trên các token trước đó
- Masked self-attention đảm bảo mỗi token chỉ attend đến các token bên trái
- Phù hợp hoàn hảo với nhiệm vụ sinh văn bản tuần tự

**b. Tối ưu cho generation tasks:**
- Không như encoder models (BERT) chỉ tốt cho understanding, decoder models tối ưu cho generation
- Có thể sinh ra văn bản dài mà vẫn giữ được ngữ cảnh
- Pre-training objective (next token prediction) trùng khớp với downstream task

**c. Hiệu quả computational:**
- Chỉ cần decode một chiều từ trái sang phải
- Không cần xử lý toàn bộ sequence như encoder-decoder models
- Inference nhanh hơn cho các tác vụ text generation

**d. Khả năng transfer learning mạnh mẽ:**
- Pre-training trên lượng lớn dữ liệu unsupervised
- Có thể fine-tune cho nhiều downstream tasks khác nhau
- Học được representations ngôn ngữ phong phú

------------------------------------------------------

## Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)

### Kết quả thực hiện

Sau khi chạy đoạn code trích xuất sentence embeddings bằng BERT, kết quả thu được như sau:

**Vector biểu diễn của câu có kích thước:** `torch.Size([1, 768])`

### Trả lời câu hỏi

#### 1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?

**Kích thước của vector biểu diễn là 768 chiều.**

Con số này tương ứng với tham số `hidden_size` của mô hình BERT.

**Giải thích chi tiết:**

Mô hình `bert-base-uncased` có cấu hình:
- **hidden_size:** 768
- **Số layer:** 12  
- **Số attention heads:** 12

Quá trình tạo vector biểu diễn:
- Mỗi token trong câu được biểu diễn bằng một vector 768 chiều
- Khi thực hiện mean pooling, chúng ta gộp thông tin từ tất cả các token thành một vector duy nhất 768 chiều

#### 2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?

Chúng ta cần sử dụng `attention_mask` vì hai lý do chính:

**a. Loại bỏ ảnh hưởng của padding tokens:**
- Khi xử lý batch nhiều câu, các câu ngắn hơn được đệm (padding) bằng token `[PAD]` để có cùng độ dài
- Các token `[PAD]` này không chứa thông tin ngữ nghĩa có ý nghĩa
- `attention_mask` giúp phân biệt token thực (value=1) và token padding (value=0)

**b. Tính toán trung bình chính xác:**
- **Không sử dụng mask:** `(tổng tất cả vectors) / (tổng số tokens)`
- **Sử dụng mask:** `(tổng vectors token thực) / (số token thực)`
- Đảm bảo chỉ các token có nghĩa được tính vào giá trị trung bình