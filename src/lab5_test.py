from sklearn.model_selection import train_test_split
from preprocessing.regex_tokenizer import RegexTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # define dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

    texts_train, texts_test, label_train, label_test = train_test_split(texts, labels, train_size=0.8, random_state=42, stratify=labels)
    print(texts_train)
    print(texts_test)

if __name__ == "__main__":
    main()