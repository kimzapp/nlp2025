import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import load_dataset
from task2 import TextClassifier


class RegexTokenizer:
    def __init__(self, pattern=r"\b\w+\b"):
        self.pattern = re.compile(pattern)
        
    def tokenize(self, text: str) -> List[str]:
        return self.pattern.findall(text.lower())
    
    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


# --- Script Đánh giá chính ---

# --- dataset ---
print("\n--- Tải dataset 'zeroshot/twitter-financial-news-sentiment' ---")
# tải dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

# dataset này có 3 nhãn: 0 (tiêu cực), 1 (tích cực), 2 (trung tính)
# chúng ta sẽ lọc để giữ lại bài toán nhị phân (tiêu cực vs. tích cực)
print("...Đang lọc bỏ các nhãn 'neutral' (nhãn == 2)...")
train_data = ds['train'].filter(lambda example: example['label'] != 2)
test_data = ds['validation'].filter(lambda example: example['label'] != 2)

# --- train test split ---
# Không cần train_test_split vì chúng ta dùng split 'train' và 'validation'
X_train = train_data['text']
y_train = train_data['label']

X_test = test_data['text']
y_test = test_data['label']

print("\n--- Dữ liệu ---")
print(f"Số mẫu Train: {len(X_train)}")
print(f"Số mẫu Test:  {len(X_test)}")


# --- Khởi tạo ---
print("\n--- Khởi tạo ---")
# tokneizer
tokenizer = RegexTokenizer(pattern=r"\b\w+\b")

# vectorizer
vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    token_pattern=None,
    stop_words="english"  # Thêm stop_words cho dataset thật
)

# classififer
classifier = TextClassifier(vectorizer=vectorizer)
print("Tokenizer, Vectorizer, và Classifier đã sẵn sàng.")

# --- Train ---
print("\n--- Training ---")
classifier.fit(X_train, y_train)
print("Training completed")

# --- Predict ---
print("\n---  Dự đoán ---")
y_pred = classifier.predict(X_test)

# In ra một vài dự đoán mẫu
print("\n--- Một vài ví dụ dự đoán ---")
for i in range(5):
    print(f"Văn bản: {X_test[i][:70]}...")
    print(f"  -> Nhãn Thật: {y_test[i]} | Nhãn Dự đoán: {y_pred[i]}")


# --- Evaluate ---
print("\n--- Kết quả Đánh giá ---")
# Tính toán các chỉ số
metrics = classifier.evaluate(y_test, y_pred)

# In các chỉ số
for key, value in metrics.items():
    print(f"{key.capitalize():<10}: {value:.2%}")