import os
import sys
from typing import Iterator, List
from gensim.models import Word2Vec

# --- Cấu hình ---
INPUT_PATH = r'F:\nlp2025\data\UD_English-EWT\en_ewt-ud-train.txt'
OUTPUT_DIR = "results"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "word2vec_ewt.model")

VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 5
WORKERS = os.cpu_count() or 1
EPOCHS = 100
SG = 1   # 1 = skip-gram, 0 = CBOW
SEED = 42
# -----------------


class UDConlluSentenceIterator:
    """
    Iterator đọc từng câu từ file CoNLL-U (hoặc text thường) theo cách tiết kiệm bộ nhớ.
    Trả về danh sách các token trong mỗi câu.
    """
    def __init__(self, filepath: str, encoding: str = "utf-8"):
        self.filepath = filepath
        self.encoding = encoding

    def __iter__(self) -> Iterator[List[str]]:
        with open(self.filepath, "r", encoding=self.encoding) as f:
            sentence_tokens = []
            is_conllu = False
            for line in f:
                line = line.strip()
                if not line:
                    if sentence_tokens:
                        yield sentence_tokens
                        sentence_tokens = []
                    continue

                # Dòng CoNLL-U có thể có tab
                if "\t" in line:
                    parts = line.split("\t")
                    if parts and parts[0].replace("-", "").isdigit():
                        if len(parts) > 1:
                            sentence_tokens.append(parts[1])
                        is_conllu = True
                    else:
                        sentence_tokens.extend(line.split())
                else:
                    if not is_conllu:
                        yield line.split()
                    else:
                        sentence_tokens.extend(line.split())

            if sentence_tokens:
                yield sentence_tokens


def main():
    if not os.path.isfile(INPUT_PATH):
        print(f"Lỗi: không tìm thấy file {INPUT_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Đang xây dựng từ vựng...")
    sentences_vocab = UDConlluSentenceIterator(INPUT_PATH)
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        seed=SEED
    )
    model.build_vocab(sentences_vocab)
    print(f"Số lượng từ trong vocab: {len(model.wv.index_to_key)}")

    print("Bắt đầu huấn luyện...")
    sentences_train = UDConlluSentenceIterator(INPUT_PATH)
    model.train(sentences_train, total_examples=model.corpus_count, epochs=EPOCHS)
    print("Huấn luyện xong.")

    print(f"Lưu model tại: {OUTPUT_MODEL}")
    model.save(OUTPUT_MODEL)
    print("Đã lưu xong model.")

    # --- Demo sử dụng model ---
    if 'game' in model.wv:
        print("\nTừ tương tự 'game':")
        for w, score in model.wv.most_similar('game', topn=10):
            print(f"  {w:<15} {score:.4f}")
    else:
        print("\n'tgame' không nằm trong vocab.")

    if {'king', 'man', 'woman'}.issubset(model.wv.key_to_index):
        print("\nPhép tương tự: king - man + woman = ?")
        for w, score in model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=5):
            print(f"  {w:<15} {score:.4f}")
    else:
        print("\nKhông đủ từ để làm ví dụ analogy (king, man, woman).")


if __name__ == "__main__":
    main()
