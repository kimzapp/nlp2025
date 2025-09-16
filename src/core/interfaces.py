from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

class BaseVectorizer(ABC):
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.vocab = None

    @abstractmethod
    def fit(self, corpus: list[str]):
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass    