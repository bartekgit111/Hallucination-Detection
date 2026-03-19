
from data_sets import Data
from w2v_model import Word2Vec
import numpy as np
#
# # 1. przygotowanie danych
# wiki = Data("data/enwik8", window_size=2, negative_size=5)
#
# wiki.prepere_data()
#
# # 2. model
# model = Word2Vec(
#     vocab_size=wiki.vocab_size,
#     embedding_dim=50,
#     lr=0.01
# )
#
# # 3. trening
# # for i, (center, context) in enumerate(wiki.pairs[:10000]):
# #
# #     negatives = wiki.generate_negative_words()
# #
# #     loss = model.train_step(center, context, negatives)
# #
# #     if i % 1000 == 0:
# #         print(f"step {i}, loss {loss}")
#
# pair_generator = wiki.generate_pairs()
#
# loss_sum = 0
#
# for i, (center, context) in enumerate(pair_generator):
#
#     negatives = wiki.generate_negative_words()
#
#     loss = model.train_step(center, context, negatives)
#
#     loss_sum += loss
#
#     if i % 1000 == 0 and i > 0:
#         print(f"step {i}, avg loss {loss_sum / 1000}")
#         loss_sum = 0
#
#     if i > 50000:   # limit na test
#         break
#
#
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# w1 = "wikipedia"
# w2 = "wiki"
#
# i1 = wiki.word_to_index[w1]
# i2 = wiki.word_to_index[w2]
#
# sim = cosine_similarity(model.W[i1], model.W[i2])
#
# print(sim)

def fit(model, data, epochs=3, print_update=10000):

    for epoch in range(epochs):

        indices = np.arange(len(data.text_as_indexes))
        np.random.shuffle(indices)

        loss_sum = 0
        step = 0

        for i in indices:

            center = data.text_as_indexes[i]

            for j in range(i - data.window_size, i + data.window_size + 1):

                if j == i or j < 0 or j >= len(data.text_as_indexes):
                    continue

                context = data.text_as_indexes[j]

                negatives = data.generate_negative_words()

                loss = model.train_step(center, context, negatives)

                loss_sum += loss
                step += 1

                if step % print_update == 0:
                    print(f"epoch {epoch}, step {step}, avg loss {loss_sum / print_update}")
                    loss_sum = 0

        print(f"END epoch {epoch}")

def train_batch(self, centers, contexts, negatives):

    W = self.W
    U = self.U

    v_c = W[centers]                 # (B, D)
    u_o = U[contexts]                # (B, D)

    # positive
    pos_scores = self.sigmoid(np.sum(v_c * u_o, axis=1))  # (B,)
    pos_grad = (pos_scores - 1)[:, None]                  # (B,1)

    # negative
    u_k = U[negatives]                                   # (B, K, D)
    neg_scores = self.sigmoid(np.sum(v_c[:, None, :] * u_k, axis=2))  # (B,K)
    neg_grad = neg_scores[:, :, None]                    # (B,K,1)

    # gradients
    grad_v = pos_grad * u_o + np.sum(neg_grad * u_k, axis=1)  # (B,D)
    grad_u_o = pos_grad * v_c                                 # (B,D)
    grad_u_k = neg_grad * v_c[:, None, :]                     # (B,K,D)

    # updates
    W[centers] -= self.lr * grad_v
    U[contexts] -= self.lr * grad_u_o

    # update negatives
    B, K = negatives.shape
    for k in range(K):
        U[negatives[:, k]] -= self.lr * grad_u_k[:, k]

    # loss (opcjonalnie)
    loss = -np.log(pos_scores + 1e-10).sum()
    loss -= np.log(1 - neg_scores + 1e-10).sum()

    return loss

def fit_batch(model, data, epochs=1, batch_size=256, print_update=10000):

    pairs = data.generate_pairs()   # (N, 2)

    for epoch in range(epochs):

        np.random.shuffle(pairs)

        loss_sum = 0

        for i in range(0, len(pairs), batch_size):

            batch = pairs[i:i + batch_size]

            centers = batch[:, 0]
            contexts = batch[:, 1]

            negatives = data.generate_negative_words(len(batch))

            loss = model.train_batch(centers, contexts, negatives)

            loss_sum += loss

            if i % (batch_size * print_update) == 0 and i > 0:
                avg_loss = loss_sum / (batch_size * print_update)
                print(f"epoch {epoch}, step {i}, avg loss {avg_loss}")
                loss_sum = 0