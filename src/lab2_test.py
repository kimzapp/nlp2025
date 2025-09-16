from representations.count_vectorizer import CountVectorizer
from preprocessing.regex_tokenizer import RegexTokenizer
from pprint import pprint

if __name__ == "__main__":
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)

    word_vectors = vectorizer.fit_transform(corpus=corpus)
    print("Vocabulary:")
    pprint(vectorizer.vocab)
    print("Word vector:")
    pprint(word_vectors)