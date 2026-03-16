# from w2v_model import Word2Vec
#
# model = Word2Vec(vocab_size=10, embedding_dim=5, lr=0.01)
#
# center = 2
# context = 4
# negatives = [1, 7, 8]
#
# loss = model.train_step(center, context, negatives)
#
# print("loss:", loss)
# print("W shape:", model.W.shape)
# print("U shape:", model.U.shape)
#
#
# model = Word2Vec(vocab_size=10, embedding_dim=10, lr=0.05)
#
# center = 2
# context = 4
# negatives = [1, 7, 8, 9]
#
# for step in range(20):
#     loss = model.train_step(center, context, negatives)
#     print(step, loss)