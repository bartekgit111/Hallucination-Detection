import numpy as np
from data_sets import Data
from w2v_model import Word2Vec
from training import fit


def cosine_distance(a, b):
    """
    This function implements cosine distance of two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_test():
    wiki = Data("data/enwik8", window_size=2, negative_size=3, max_tokens=None)
    wiki.prepere_data()

    print(f"vocab size: {wiki.vocab_size}")
    print(f"num tokens: {len(wiki.text_as_indexes)}")

    model = Word2Vec(vocab_size=wiki.vocab_size, embedding_dim=50, lr=0.01)

    fit(model, wiki, epochs=1, print_update=100000)

    id1 = wiki.word_to_index["wikipedia"]
    id2 = wiki.word_to_index["wiki"]
    distance = cosine_distance(model.W[id1], model.W[id2])

    print(f"wikipedia & wiki - similarity: {distance:.4f}")


run_test()
