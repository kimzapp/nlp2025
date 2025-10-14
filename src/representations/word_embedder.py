import gensim.downloader as api
from preprocessing.simple_tokenizer import SimpleTokenizer
import numpy as np

class WordEmbedder:
    def __init__(self, model_name: str='glove-wiki-gigaword-50'):
        self.model = api.load(model_name)

    def get_vector(self, word: str):
        if word in self.model:
            return self.model[word]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")
        
    def get_similarity(self, word1: str, word2: str):
        vector1 = self.get_vector(word1)
        vector2 = self.get_vector(word2)
        norm1 = sum(x * x for x in vector1) ** 0.5
        norm2 = sum(x * x for x in vector2) ** 0.5
        dot_product = sum(x * y for x, y in zip(vector1, vector2))
        return dot_product / (norm1 * norm2)

    def get_most_similar(self, word: str, topn: int = 10):
        if word in self.model:
            return self.model.most_similar(word, topn=topn)
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")    
        
    def embed_document(self, document: str):
        tokenizer = SimpleTokenizer()
        words = tokenizer.tokenize(document)
        doc_vector = np.zeros(self.model.vector_size)
        for word in words:
            try:
                doc_vector += self.get_vector(word)
            except:
                # ignore OOV words
                continue
        
        return doc_vector / len(words) if words else doc_vector # lấy trung bình