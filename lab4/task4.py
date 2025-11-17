import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from xgboost import XGBClassifier

# --- Tải tài nguyên NLTK ---
print("Đang tải tài nguyên NLTK (punkt, stopwords)...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
print("Tải xong tài nguyên NLTK.")
# ---------------------------------------------------


# Đường dẫn đến tệp GloVe 100 chiều
GLOVE_PATH = r'F:\nlp2025\wiki_giga_2024_100.txt'
EMBEDDING_DIM = 100 # Phải khớp với tệp GloVe

# Tải danh sách stopwords tiếng Anh
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("LỖI: Chưa tải tài nguyên stopwords của NLTK.")
    print("Vui lòng bỏ comment 2 dòng nltk.download(...) ở đầu code và chạy lại.")
    exit()

def load_glove_embeddings(glove_file):
    print(f"Đang tải GloVe embeddings từ {glove_file} (bằng Python)...")
    glove_dict = {}
    
    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                try:
                    # Chuyển đổi các giá trị embedding sang numpy array
                    vector = np.array([float(val) for val in parts[1:]])
                    
                    if len(vector) == EMBEDDING_DIM:
                        glove_dict[word] = vector
                except ValueError:
                    continue

        print(f"Đã tải {len(glove_dict)} word vectors.")
        return glove_dict

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp GloVe tại: {glove_file}")
        return None
    except Exception as e:
        print(f"LỖI khi đang đọc tệp GloVe: {e}")
        return None

def preprocess_text(text):
    # 1. Lowercase và Regex (chỉ giữ lại chữ cái và khoảng trắng)
    text = re.sub(r'[^a-z\s]', '', str(text).lower())
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove Stopwords
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return filtered_tokens

def get_average_vector(words, glove_model, dim):
    """
    Hàm tính trung bình vector GloVe cho một danh sách các từ.
    Tương đương với UDF 'vector_averager' của bạn.
    """
    word_vectors = []
    for word in words:
        if word in glove_model:
            word_vectors.append(glove_model[word])
    
    if not word_vectors:
        # Nếu không từ nào có trong GloVe, trả về vector 0
        return np.zeros(dim)
    
    # Tính trung bình các vector
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector

def main():
    
    # 1. [CẢI TIẾN] Tải GloVe Embeddings
    glove_map = load_glove_embeddings(GLOVE_PATH)
    if glove_map is None:
        print("Không thể tải GloVe. Dừng chương trình.")
        return

    # Xác định đường dẫn tương đối đến tệp dữ liệu
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Xảy ra khi chạy trong môi trường tương tác (ví dụ: Jupyter)
        base_dir = os.getcwd()
        
    data_path = os.path.join(base_dir, "data", "sentiments.csv")

    try:
        # 2. Tải và chuẩn bị dữ liệu (dùng Pandas)
        print(f"Đang tải dữ liệu từ {data_path}...")
        df = pd.read_csv(data_path)
        
        # Xử lý NA
        df = df.dropna(subset=["sentiment", "text"])
        
        df["label"] = ((df["sentiment"].astype(int) + 1) / 2).astype(int)

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp dữ liệu tại {data_path}")
        return
    except Exception as e:
        print(f"LỖI khi đọc dữ liệu: {e}")
        return

    print("Dữ liệu đã được tải và tạo label.")
    print(df.head())

    # 3. Tiền xử lý (Tokenizer + StopWords)
    print("Đang tiền xử lý văn bản (tokenize, remove stopwords)...")
    df["filtered_words"] = df["text"].apply(preprocess_text)
    
    print("Dữ liệu sau khi tiền xử lý (Tokenizer + StopWords):")
    print(df[["text", "filtered_words"]].head())

    # 4. [CẢI TIẾN] - Áp dụng GloVe để tạo features
    print("Đang tạo features (vector trung bình GloVe)...")
    df["features"] = df["filtered_words"].apply(
        lambda words: get_average_vector(words, glove_map, EMBEDDING_DIM)
    )

    print("Dữ liệu sau khi áp dụng GloVe (cột 'features'):")
    print(df[["label", "features"]].head())

    # 5. Chuẩn bị dữ liệu
    X = np.vstack(df["features"].values)
    y = df["label"].values

    print(f"Hoàn tất tạo ma trận features X (shape: {X.shape}) và vector y (shape: {y.shape})")

    # 6. Huấn luyện và Đánh giá mô hình
    
    # Chia dữ liệu
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, 
        test_size=0.2,    # 80% train, 20% test
        random_state=42, 
        stratify=y        
    )

    # Sử dụng classifier mạnh hơn: XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    print("Bắt đầu huấn luyện XGBoost (Scikit-learn)...")
    xgb_model = xgb.fit(X_train, y_train)
    print("Huấn luyện hoàn tất.")

    # Đánh giá trên tập test
    predictions = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Set Accuracy (GloVe) = {accuracy * 100:.2f}%")

    f1 = f1_score(y_test, predictions, average='binary') 
    print(f"Test Set F1 Score (GloVe) = {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=["Negative (0)", "Positive (1)"]))

if __name__ == "__main__":
    main()