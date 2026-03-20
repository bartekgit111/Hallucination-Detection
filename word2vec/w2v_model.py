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

    # def train_step(self, center, context, negatives):
    #     v_c = self.W[center]
    #     u_o = self.U[context].copy()
    #
    #     positive_score = self.sigmoid(np.dot(v_c, u_o))
    #
    #     loss = -np.log(positive_score + 1e-15)
    #
    #     positive_grad = positive_score - 1
    #
    #     self.W[center] -= self.lr * positive_grad * u_o
    #     self.U[context] -= self.lr * positive_grad * v_c
    #
    #     for neg in negatives:
    #         u_k = self.U[neg]
    #
    #         negative_score = self.sigmoid(np.dot(v_c, u_k))
    #
    #         loss -= np.log(1 - negative_score + 1e-15)
    #
    #         negative_grad = negative_score
    #
    #         self.W[center] -= self.lr * negative_grad * u_k
    #         self.U[neg] -= self.lr * negative_grad * v_c
    #
    #     return loss

    def train_step(self, center, context, negatives):
        v_c = self.W[center]
        u_o = self.U[context]

        # positive
        score_pos = self.sigmoid(np.dot(v_c, u_o))
        grad_pos = score_pos - 1

        grad_v = grad_pos * u_o
        grad_u_o = grad_pos * v_c

        loss = -np.log(score_pos + 1e-15)

        # negatives
        for neg in negatives:
            u_k = self.U[neg]

            score_neg = self.sigmoid(np.dot(v_c, u_k))
            grad_neg = score_neg

            grad_v += grad_neg * u_k
            self.U[neg] -= self.lr * grad_neg * v_c

            loss -= np.log(1 - score_neg + 1e-15)

        # update na końcu
        self.W[center] -= self.lr * grad_v
        self.U[context] -= self.lr * grad_u_o

        return loss
