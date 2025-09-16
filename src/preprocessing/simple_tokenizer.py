from core.interfaces import BaseTokenizer

class SimpleTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()