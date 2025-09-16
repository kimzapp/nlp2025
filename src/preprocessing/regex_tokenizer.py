from core.interfaces import BaseTokenizer
import re

class RegexTokenizer(BaseTokenizer):
    def __init__(self, pattern: str = r'\w+|[^\w\s]'):
        super().__init__()
        self.pattern = pattern

    def tokenize(self, text: str) -> list[str]:
        return re.findall(self.pattern, text.lower())