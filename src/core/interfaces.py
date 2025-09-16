from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass
