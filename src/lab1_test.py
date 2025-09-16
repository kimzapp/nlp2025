from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import RegexTokenizer
from core.dataset_loaders import load_raw_text_data

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

    dataset_path = r"E:\nlp2025\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500] # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
