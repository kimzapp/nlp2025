from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

class TextClassfier:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=10000)

    def fit(self, texts, labels):
        tokenized = self.vectorizer.fit_transform(texts)
        self._model.fit(tokenized, labels)

    def predict(self, texts):
        tokenized = self.vectorizer.transform(texts)
        pred = self._model.predict(tokenized)
        return pred
    
    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
            'precision': precision_score(y_true=y_true, y_pred=y_pred),
            'recall': recall_score(y_true=y_true, y_pred=y_pred)
        }