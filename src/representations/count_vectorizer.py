from core.interfaces import BaseVectorizer, BaseTokenizer

class CountVectorizer(BaseVectorizer):
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__(tokenizer=tokenizer)
        
    def fit(self, corpus: list[str]):
        vocabulary = set()
        for document in corpus:
            vocabulary.update(self.tokenizer.tokenize(document))
        vocabulary = sorted(vocabulary)
        self.vocab = {w:i for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

    def transform(self, documents: list[str]) -> list[list[int]]:
        results = []
        for document in documents:
            tokens = self.tokenizer.tokenize(document)
            word_vector = [0]*self.vocab_size
            for token in tokens:
                word_vector[self.vocab[token]] += 1
            results.append(word_vector)
        return results


    def fit_transform(self, corpus: list[str]) -> list[list[str]]:
        self.fit(corpus=corpus)
        return self.transform(corpus)           
            