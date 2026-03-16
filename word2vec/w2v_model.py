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
        # v_c = self.W[center].copy()
        # u_o = self.U[context].copy()

        v_c = self.W[center]
        u_o = self.U[context]

        positive_score = self.sigmoid(np.dot(v_c, u_o))

        loss = -np.log(positive_score)

        positive_grad = positive_score - 1  # dlaczego gradient jest taki?

        self.W[center] -= self.lr * positive_grad * u_o  # dlaczego tak aktualizujemy
        self.U[context] -= self.lr * positive_grad * v_c  # dlaczego tak aktualizujemy

        for neg in negatives:
            u_k = self.U[neg]

            negative_score = self.sigmoid(np.dot(v_c, u_k))

            loss -= np.log(1 - negative_score)

            negative_grad = negative_score

            self.W[center] -= self.lr * negative_grad * u_k
            self.U[neg] -= self.lr * negative_grad * v_c

        return loss
