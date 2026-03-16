import re
from collections import Counter
import numpy as np

class data():
    def __init__(self, path, window_size = 3, negative_size = 5):
        self.path = path
        self.raw_text = ""
        self.words = []
        self.vocab = []
        self.vocab_size = 0
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()
        self.text_as_indexes = []
        self.window_size = window_size
        self.pairs = []
        self.negative_size = negative_size

    def raw_text_from_file(self):
        with open(self.path, "r", encoding="utf8") as file:
            self.raw_text = file.read()

    def tokenization(self):
        self.raw_text = re.sub(r"<[^>]+>", " ", self.raw_text)
        self.raw_text = self.raw_text.lower()
        self.raw_text = re.sub(r"[^a-z\s]", " ", self.raw_text)
        self.raw_text = re.sub(r"\s+", " ", self.raw_text)

        self.words  = self.raw_text.split()

    def build_vocab(self):
        self.word_counts = Counter(self.words)
        self.vocab = list(self.word_counts.keys())
        self.vocab_size = len(self.vocab)

        self.word_to_index = {
            word: idx for idx, word in enumerate(self.vocab)
        }

        self.index_to_word = {
            idx: word for idx, word in enumerate(self.vocab)
        }

    def text_to_indexes(self):
        self.text_as_indexes = [self.word_to_index[word] for word in self.words]

    def generate_pairs(self):

        pairs = []

        for i in range(len(self.text_as_indexes)):

            center_word = self.text_as_indexes[i]

            for j in range(i - self.window_size, i + self.window_size + 1):

                if j != i and 0 <= j < len(self.text_as_indexes):
                    context_word = self.text_as_indexes[j]

                    pairs.append((center_word, context_word))

        self.pairs = pairs

    def generate_negative_words(self):
        negative_words = np.random.choice(self.vocab_size, size=self.negative_size)

        return negative_words





# wiki = data("data\enwik8")
# wiki.raw_text_from_file()
# wiki.tokenization()
# wiki.build_vocab()
# wiki.text_to_indexes()
# wiki.generate_pairs()

# print(len(wiki.words))
# print(wiki.words[:10])
#
# print(wiki.vocab_size)
# print(wiki.vocab[:10])
#
# print(list(wiki.word_to_index.items())[:10])
# print(list(wiki.index_to_word.items())[:10])
# print(list(wiki.text_as_indexes)[:10])
# print(wiki.pairs[:10])

# neg = wiki.generate_negative_words()
#
# print([wiki.index_to_word[i] for i in neg])
