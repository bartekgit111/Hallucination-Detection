import numpy as np


class Word2Vec:

    def __init__(self, vocab_size, embedding_dim=100, lr=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr

        self.W = np.random.uniform(-1.0, 1.0, size=(self.vocab_size, self.embedding_dim))
        self.U = np.random.uniform(-1.0, 1.0, size=(self.vocab_size, self.embedding_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_step(self, center, context, negatives):
        v_c = self.W[center].copy()
        u_o = self.U[context].copy()

        # v_c = self.W[center]
        # u_o = self.U[context]

        positive_score = self.sigmoid(np.dot(v_c, u_o))

        loss = -np.log(positive_score + 1e-15)

        positive_grad = positive_score - 1  # dlaczego gradient jest taki?

        self.W[center] -= self.lr * positive_grad * u_o  # dlaczego tak aktualizujemy
        self.U[context] -= self.lr * positive_grad * v_c  # dlaczego tak aktualizujemy

        for neg in negatives:
            u_k = self.U[neg]

            negative_score = self.sigmoid(np.dot(v_c, u_k))

            loss -= np.log(1 - negative_score + 1e-15)

            negative_grad = negative_score

            self.W[center] -= self.lr * negative_grad * u_k
            self.U[neg] -= self.lr * negative_grad * v_c

        return loss

    def train_batch(self, centers, contexts, negatives):

        W = self.W
        U = self.U

        v_c = W[centers]  # (B, D)
        u_o = U[contexts]  # (B, D)

        # positive
        pos_scores = self.sigmoid(np.sum(v_c * u_o, axis=1))  # (B,)
        pos_grad = (pos_scores - 1)[:, None]  # (B,1)

        # negative
        u_k = U[negatives]  # (B, K, D)
        neg_scores = self.sigmoid(np.sum(v_c[:, None, :] * u_k, axis=2))  # (B,K)
        neg_grad = neg_scores[:, :, None]  # (B,K,1)

        # gradients
        grad_v = pos_grad * u_o + np.sum(neg_grad * u_k, axis=1)  # (B,D)
        grad_u_o = pos_grad * v_c  # (B,D)
        grad_u_k = neg_grad * v_c[:, None, :]  # (B,K,D)

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
