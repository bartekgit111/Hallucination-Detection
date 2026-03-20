import numpy as np
from data_sets import Data
from w2v_model import Word2Vec
from training import fit


def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_test():
    wiki = Data("data/enwik8", window_size=2, negative_size=3, max_tokens=500000)
    wiki.prepere_data()

    print(f"vocab size: {wiki.vocab_size}")
    print(f"num tokens: {len(wiki.text_as_indexes)}")

    model = Word2Vec(vocab_size=wiki.vocab_size, embedding_dim=50, lr=0.01)

    fit(model, wiki, epochs=1, print_update=10000)

    test_pairs = [
        ("wikipedia", "wiki"),
        ("king", "queen"),
        ("man", "woman"),
    ]

    for word1, word2 in test_pairs:
        if word1 in wiki.word_to_index and word2 in wiki.word_to_index:
            id1 = wiki.word_to_index[word1]
            id2 = wiki.word_to_index[word2]

            distance = cosine_distance(model.W[id1], model.W[id2])
            print(f"{word1} vs {word2} -> similarity: {distance:.4f}")
        else:
            print(f"{word1} or {word2} not in vocab")


run_test()
