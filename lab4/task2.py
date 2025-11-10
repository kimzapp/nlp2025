from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# Tạo một type hint chung cho các vectorizer của scikit-learn
Vectorizer = Union[TfidfVectorizer, CountVectorizer]

class TextClassifier: 
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        # _model sẽ được gán sau khi fit() được gọi
        self._model: LogisticRegression = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Args:
            texts: Danh sách các đoạn văn bản huấn luyện.
            labels: Danh sách các nhãn tương ứng.
        """
        # 1. Vectorize văn bản huấn luyện
        # Dùng fit_transform để học từ vựng và biến đổi
        X = self.vectorizer.fit_transform(texts)
        
        # 2. Khởi tạo và huấn luyện mô hình
        # Sử dụng solver='liblinear' tốt cho dataset nhỏ
        self._model = LogisticRegression(solver='liblinear')
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Args:
            texts: Danh sách các đoạn văn bản cần dự đoán.
            
        Returns:
            Danh sách các nhãn dự đoán.
        """
        if self._model is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi .fit() trước.")
            
        # 1. Vectorize văn bản mới
        # Dùng transform để sử dụng từ vựng đã học
        X = self.vectorizer.transform(texts)
        
        # 2. Dự đoán
        predictions = self._model.predict(X)
        
        # Trả về dưới dạng list
        return predictions.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Args:
            y_true: Danh sách các nhãn thật.
            y_pred: Danh sách các nhãn dự đoán.
            
        Returns:
            Một dictionary chứa các chỉ số: accuracy, precision, recall, f1_score.
        """
        # Tính toán các chỉ số
        # Thêm zero_division=0 để tránh lỗi/warning khi một lớp không có dự đoán
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Trả về dưới dạng dictionary
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        return metrics