import numpy as np


class Word2Vec:
    """
    This class implements Skip-gram Word2Vec model with negative sampling.

    learns:
    - Input embeddings (W)
    - Output embeddings (U)

    training with sgd

    W : input embeddings (vocab_size x embedding_dim)
    U : output embeddings (vocab_size x embedding_dim)
    """

    def __init__(self, vocab_size, embedding_dim=100, lr=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr

        self.W = np.random.uniform(-1.0, 1.0, size=(self.vocab_size, self.embedding_dim))
        self.U = np.random.uniform(-1.0, 1.0, size=(self.vocab_size, self.embedding_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_step(self, center, context, negatives):
        """
        this method performs one SGD update for a single (center, context) pair.


        center : index of center word.
        context : index of positive context word.
        negatives : indexes of negative samples.

        returns
        - loss value

        updates:
        - W[center]
        - U[context]
        - U[negatives]

        """
        v_c = self.W[center]
        u_o = self.U[context]

        # positive
        score_pos = self.sigmoid(np.dot(v_c, u_o))
        grad_pos = score_pos - 1

        grad_v = grad_pos * u_o
        grad_u_o = grad_pos * v_c

        loss = -np.log(score_pos + 1e-15)


        for neg in negatives:
            u_k = self.U[neg]

            score_neg = self.sigmoid(np.dot(v_c, u_k))
            grad_neg = score_neg

            grad_v += grad_neg * u_k
            self.U[neg] -= self.lr * grad_neg * v_c

            loss -= np.log(1 - score_neg + 1e-15)


        self.W[center] -= self.lr * grad_v
        self.U[context] -= self.lr * grad_u_o

        return loss
