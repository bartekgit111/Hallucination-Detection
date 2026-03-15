import re
from collections import Counter

class data():
    def __init__(self, path):
        self.path = path
        self.raw_text = ""
        self.words = []
        self.vocab = []
        self.vocab_size = 0
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()

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



wiki = data("data\enwik8")
wiki.raw_text_from_file()
wiki.tokenization()
wiki.build_vocab()

print(len(wiki.words))
print(wiki.words[:10])

print(wiki.vocab_size)
print(wiki.vocab[:10])

print(list(wiki.word_to_index.items())[:10])
print(list(wiki.index_to_word.items())[:10])
print(list(wiki.index_to_word.items())[:1])
