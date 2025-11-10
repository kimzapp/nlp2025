import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Dataset
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = np.array([1, 0, 1, 0, 1, 0]) # 1 for positive, 0 for negative

# 2. Vectorize
# Khởi tạo TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit (huấn luyện) và transform (biến đổi) văn bản
X = vectorizer.fit_transform(texts)

# In kết quả để kiểm tra
print("--- Dữ liệu văn bản gốc ---")
print(texts)

print("\n--- Nhãn ---")
print(labels)

print("\n--- Ma trận TF-IDF (Shape: Mẫu tin x Đặc trưng) ---")
print(X.shape)

print("\n--- Ma trận TF-IDF (Dạng thưa) ---")
print(X)

print("\n--- Ma trận TF-IDF (Dạng đầy đủ) ---")
print(X.toarray())

print("\n--- Tên các đặc trưng (từ vựng) ---")
print(vectorizer.get_feature_names_out())