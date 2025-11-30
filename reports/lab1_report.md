## Lab 1: Word tokenization 

#### Task 1: Implement Simple Tokenizer (Tokenize sentences base on white space)

Output:
```
Text1:  Hello, world! This is a test.

Result: ['hello,', 'world!', 'this', 'is', 'a', 'test.']
```

```
Text2:  NLP is fascinating... isn't it?

Result: ['nlp', 'is', 'fascinating...', "isn't", 'it?']

```

```
Text3:  Let's see how it handles 123 numbers and punctuation!

Result: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation!']

```

#### Task 2: Implement Regex-based Tokenizer

Tokenize senteces using regular expression

Output:

```
Text1:  Hello, world! This is a test.

Result: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
```

```
Text2:  NLP is fascinating... isn't it?

Result: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
```

```
Text3:  Let's see how it handles 123 numbers and punctuation!

Result: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

#### Task 3: Tokenization with UD_English-EWT Dataset

Implement a method for loading raw data, then tokenize them using pre-defined tokenizer

```
Original Sample: 

"Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ..."
```

```
SimpleTokenizer Output (first 20 tokens): 
['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani,', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim,', 'near', 'the']

RegexTokenizer Output (first 20 tokens): 
['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

## Lab 2: Vectorization

- Implement abstract class for vectorizer and CountVectorizer class (BOW method).
- Evaluate on some sample data

```
Sample corpus:
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
```

Result:

```
Vocabulary:
{
    '.': 0,
    'a': 1,
    'ai': 2,
    'i': 3,
    'is': 4,
    'love': 5,
    'nlp': 6,
    'of': 7,
    'programming': 8,
    'subfield': 9
}

Word vectors:
    "I love NLP." -> [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    "I love programming." -> [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    "NLP is a subfield of AI." -> [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

```
