import numpy as np
from data_sets import Data
from w2v_model import Word2Vec
from training import fit,  fit_batch



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_test():

    print("=== LOADING DATA ===")

    data = Data("data/enwik8", window_size=2, negative_size=2)
    data.prepere_data()


    print(f"vocab size: {data.vocab_size}")
    print(f"num tokens: {len(data.text_as_indexes)}")

    print("\n=== INIT MODEL ===")

    model = Word2Vec(
        vocab_size=data.vocab_size,
        embedding_dim=50,
        lr=0.01
    )

    print("\n=== TRAINING ===")

    fit(model, data, epochs=1, print_update=5000)

    print("\n=== TEST SIMILARITY ===")

    test_pairs = [
        ("wikipedia", "wiki"),
        ("king", "queen"),
        ("man", "woman"),
    ]

    for w1, w2 in test_pairs:
        if w1 in data.word_to_index and w2 in data.word_to_index:
            i1 = data.word_to_index[w1]
            i2 = data.word_to_index[w2]

            sim = cosine_similarity(model.W[i1], model.W[i2])
            print(f"{w1} vs {w2} -> similarity: {sim:.4f}")
        else:
            print(f"{w1} or {w2} not in vocab")


def run_test_batch():

    print("=== LOADING DATA ===")

    data = Data("data/enwik8", window_size=2, negative_size=5)
    data.prepere_data()

    data.build_unigram_table()


    print(f"vocab size: {data.vocab_size}")
    print(f"num tokens: {len(data.text_as_indexes)}")

    print("\n=== INIT MODEL ===")

    model = Word2Vec(
        vocab_size=data.vocab_size,
        embedding_dim=50,
        lr=0.01
    )

    print("\n=== TRAINING ===")

    fit_batch(
        model,
        data,
        epochs=1,
        batch_size=256,
        print_update=100000
    )

    print("\n=== TEST ===")

    w1 = "wikipedia"
    w2 = "wiki"

    if w1 in data.word_to_index and w2 in data.word_to_index:
        i1 = data.word_to_index[w1]
        i2 = data.word_to_index[w2]

        sim = cosine_similarity(model.W[i1], model.W[i2])
        print(f"{w1} vs {w2} -> similarity: {sim:.4f}")
    else:
        print("words not in vocab")


run_test_batch()
