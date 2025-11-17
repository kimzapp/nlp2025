# Báo cáo: Xây dựng và Đánh giá Mô hình Phân loại Cảm xúc Tin tức Tài chính
---

### 1. Các công việc đã thực hiện

Quy trình được thực hiện qua các bước chính sau:

#### 1.1. Xây dựng Lõi mô hình (Task 2)
* Tạo file `src/models/text_classifier.py`.
* Implement lớp `TextClassifier` có khả năng:
    * Nhận một đối tượng `Vectorizer` (ví dụ: `TfidfVectorizer`) khi khởi tạo.
    * **Phương thức `fit()`:** Nhận `texts` và `labels`, sau đó thực hiện `vectorizer.fit_transform` và huấn luyện (`model.fit`) mô hình `LogisticRegression`.
    * **Phương thức `predict()`:** Nhận `texts` mới, thực hiện `vectorizer.transform` và trả về dự đoán từ mô hình đã huấn luyện.
    * **Phương thức `evaluate()`:** Nhận `y_true` và `y_pred`, tính toán và trả về một dictionary chứa các chỉ số (Accuracy, Precision, Recall, F1).

#### 1.2. Chuẩn bị Dữ liệu (Task 3 - Phần 1)
* Sử dụng thư viện `datasets` của Hugging Face để tải bộ dữ liệu `zeroshot/twitter-financial-news-sentiment`.
* **Tiền xử lý:** Bộ dữ liệu gốc có 3 nhãn: Tích cực (1), Tiêu cực (0), và Trung tính (2). Để phù hợp với bài toán phân loại nhị phân, chúng ta đã **lọc bỏ các mẫu tin có nhãn "trung tính" (label == 2)**.
* **Phân chia dữ liệu:** Sử dụng split `train` có sẵn để huấn luyện và split `validation` để kiểm thử (thay cho `train_test_split`).

#### 1.3. Cài đặt Pipeline và Đánh giá (Task 3 - Phần 2)
* Tạo file `test/lab5_test.py` để thực thi toàn bộ pipeline.
* **Tokenizer:** Khởi tạo `RegexTokenizer` tùy chỉnh.
* **Vectorizer:** Khởi tạo `TfidfVectorizer` (từ `scikit-learn`), truyền `RegexTokenizer` làm tham số `tokenizer` và bổ sung `stop_words='english'` để cải thiện chất lượng đặc trưng.
* **Huấn luyện và Đánh giá:**
    1.  Khởi tạo `TextClassifier` với `TfidfVectorizer`.
    2.  Huấn luyện mô hình (`classifier.fit`) trên tập `train` (X\_train, y\_train).
    3.  Thực hiện dự đoán (`classifier.predict`) trên tập `validation` (X\_test).
    4.  Tính toán và in ra kết quả đánh giá.

---

### 2. Kết quả và Phân tích

Mô hình đã được huấn luyện và đánh giá trên tập `validation`. Kết quả thu được như sau:

#### 2.1. Các chỉ số hiệu suất
* **Accuracy:** **80.41%**
* **Precision:** **78.75%**
* **Recall:** **90.53%**
* **F1-score:** **84.23%**

#### 2.2. Phân tích các chỉ số
Kết quả này rất tích cực đối với một mô hình baseline (Logistic Regression + TF-IDF):

* **Accuracy (80.41%):** Mô hình dự đoán đúng tổng thể 8 trên 10 trường hợp.
* **F1-score (84.23%):** Đây là chỉ số cân bằng giữa Precision và Recall. Mức 84.23% cho thấy mô hình hoạt động mạnh mẽ và cân bằng tốt.
* **Precision (78.75%) vs. Recall (90.53%):**
    * **Recall cao (90.53%)** là điểm nổi bật nhất. Điều này có nghĩa là mô hình **rất giỏi trong việc "bắt" (tìm ra) các tin tức tích cực (positive)**. Nó đã tìm thấy 90.53% trong tổng số tất cả các tin tích cực có trong tập test.
    * **Precision (78.75%)** thấp hơn một chút. Điều này có nghĩa là khi mô hình dự đoán một tin là "tích cực", nó dự đoán đúng 78.75% số lần. 21.25% còn lại là các tin tiêu cực bị dự đoán nhầm là tích cực.
    * **Kết luận:** Mô hình có xu hướng thiên về "thà bắt nhầm còn hơn bỏ sót" (ưu tiên Recall).

#### 2.3. Phân tích ví dụ dự đoán
Các ví dụ được cung cấp đều là các tin tức tiêu cực (nhãn thật: 0) và mô hình đã dự đoán chính xác tất cả:
- Văn bản: $ALLY - Ally Financial pulls outlook...   -> Nhãn Thật: 0 | Nhãn Dự đoán: 0

- Văn bản: $DELL $HPE - Dell, HPE targets trimmed on compute headwinds...   -> Nhãn Thật: 0 | Nhãn Dự đoán: 0

- Văn bản: $PRTY - Moody's turns negative on Party City...   -> Nhãn Thật: 0 | Nhãn Dự đoán: 0

- Văn bản: $SAN: Deutsche Bank cuts to Hold...   -> Nhãn Thật: 0 | Nhãn Dự đoán: 0

- Văn bản: $SITC: Compass Point cuts to Sell...   -> Nhãn Thật: 0 | Nhãn Dự đoán: 0

### 3. Thử nghiệm Nâng cao (PySpark) và Cải tiến Mô hình (Task 4)

Sau khi đánh giá mô hình `scikit-learn`, chúng ta tiếp tục với ví dụ nâng cao sử dụng **Apache PySpark** để xử lý các bộ dữ liệu lớn hơn và thử nghiệm các phương pháp cải tiến mô hình như đã nêu trong Task 4.

#### 3.1. Kết quả Baseline (PySpark + TF-IDF + Logistic Regression)

Đầu tiên, chúng ta chạy pipeline baseline trên PySpark. Pipeline này tương tự như phiên bản `scikit-learn`, bao gồm:
1.  `Tokenizer` (Tách từ)
2.  `StopWordsRemover` (Loại bỏ stop words)
3.  `HashingTF` (Tương đương CountVectorizer)
4.  `IDF` (Tính trọng số TF-IDF)
5.  `LogisticRegression` (Mô hình)

Mô hình baseline này đạt được kết quả khá tốt, làm nền tảng để so sánh:

* **Test Set Accuracy:** **72.95%**
* **Test Set F1 Score (Tổng thể):** **0.7266**

#### 3.2. Kết quả Cải tiến (Pre-trained GloVe + XGBoost)

Theo gợi ý của Task 4, chúng ta đã thử nghiệm một pipeline nâng cao hơn, kết hợp hai phương pháp:
1.  **Advanced Embedding Methods**: Sử dụng pre-trained **GloVe** để tạo vector đặc trưng (thay thế cho TF-IDF).
2.  **More Complex Model Architectures**: Thay thế `LogisticRegression` bằng **XGBoost Classifier** (một dạng Gradient-Boosted Trees - GBTs).

Kết quả của mô hình cải tiến này như sau:

* **Test Set Accuracy (GloVe):** **71.96%**
* **Test Set F1 Score (cho lớp Positive):** **0.8036**

**Classification Report chi tiết:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Negative (0) | 0.70 | 0.40 | 0.51 | 421 |
| Positive (1) | 0.73 | 0.90 | 0.80 | 738 |
| | | | | |
| **Accuracy** | | | **0.72** | **1159** |
| Macro Avg | 0.71 | 0.65 | 0.66 | 1159 |
| Weighted Avg | 0.72 | 0.72 | 0.70 | 1159 |

#### 3.3. Phân tích và So sánh

1.  **Về Accuracy và F1 Tổng thể (Weighted Avg):**
    Một cách bất ngờ, tổng thể Accuracy (71.96% vs 72.95%) và Weighted Avg F1-score (0.70 vs 0.7266) của mô hình nâng cao (GloVe + XGBoost) lại **thấp hơn** mô hình baseline (TF-IDF + LR).

2.  **Phân tích chi tiết (Lý do):**
    Lý do nằm ở sự **mất cân bằng nghiêm trọng** trong khả năng dự đoán của mô hình nâng cao:
    * **Điểm mạnh (Lớp Positive):** Mô hình này cực kỳ giỏi trong việc xác định sentiment **Tích cực (Positive - 1)**. Nó đạt Recall 0.90 (tìm thấy 90% các tin tích cực) và F1-score là 0.80.
    * **Điểm yếu (Lớp Negative):** Tuy nhiên, nó đã "hy sinh" khả năng nhận diện sentiment **Tiêu cực (Negative - 0)**. Recall chỉ là 0.40 (bỏ sót 60% các tin tiêu cực) và F1-score chỉ đạt 0.51.

3.  **So sánh với Baseline:**
    Mô hình baseline (TF-IDF + Logistic Regression) tuy đơn giản hơn nhưng lại cho kết quả **cân bằng và ổn định hơn** giữa hai lớp (F1 tổng thể 0.7266), trong khi mô hình GloVe + XGBoost bị thiên vị (bias) nặng về việc dự đoán lớp Positive (lớp đa số, với `support` = 738 so với 421).

### 4. Kết luận Chung từ các Thử nghiệm

* **Baseline rất quan trọng:** Cả hai mô hình baseline (TF-IDF + Logistic Regression) trên `scikit-learn` (F1: 84.23%) và `PySpark` (F1: 72.66%) đều cho thấy hiệu suất mạnh mẽ, ổn định.
* **Mô hình phức tạp không phải lúc nào cũng tốt hơn:** Việc áp dụng các kỹ thuật nâng cao như GloVe và XGBoost không đảm bảo cải thiện hiệu suất tổng thể. Trong trường hợp này, nó dẫn đến một mô hình mất cân bằng, hoạt động tốt trên một lớp nhưng kém đi đáng kể trên lớp còn lại.
* **Vấn đề nằm ở đâu:** Hiệu suất kém của mô hình GloVe + XGBoost có thể đến từ:
    * Cách vector hóa (ví dụ: chỉ lấy trung bình các vector GloVe làm mất nhiều thông tin ngữ cảnh).
    * Dữ liệu mất cân bằng (lớp Positive nhiều gấp 1.75 lần lớp Negative).
    * Chưa tinh chỉnh (tune) hyperparameter cho XGBoost.