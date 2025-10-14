from representations.word_embedder import WordEmbedder

embedder = WordEmbedder()
print("King vector:", embedder.get_vector("king"))
print("Similarity between 'king' and 'queen':", embedder.get_similarity("king", "queen"))
print("Similarity between 'king' and 'man':", embedder.get_similarity("king", "man"))
print("Most 10 similar words to 'computer':", embedder.get_most_similar("computer", topn=10))

sequence = "The queen rules the country."
print("Input sequence:", sequence)
print("Document embedding:", embedder.embed_document(sequence))