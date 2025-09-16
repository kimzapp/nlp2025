from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import RegexTokenizer


if __name__ == "__main__":
    text1 = "Hello, world! This is a test."
    text2 = "NLP is fascinating... isn't it?"
    text3 = "Let's see how it handles 123 numbers and punctuation!"
    
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    print("Text1: ", text1)
    print("Simple Tokenizer:", simple_tokenizer.tokenize(text1))
    print("Regex Tokenizer:", regex_tokenizer.tokenize(text1))

    print("\nText2: ", text2)
    print("Simple Tokenizer:", simple_tokenizer.tokenize(text2))
    print("Regex Tokenizer:", regex_tokenizer.tokenize(text2))
   
    print("\nText3: ", text3)
    print("Simple Tokenizer:", simple_tokenizer.tokenize(text3))
    print("Regex Tokenizer:", regex_tokenizer.tokenize(text3))