import re
from collections import Counter
import numpy as np


class Data:
    """
    This class does text preprocessing for training Word2Vec
    with skip-gram and negative sampling.

    - load data text
    - remove any unnecessary fragments of text
    - tokenize and build vocabulary
    - convert words to indexes
    - remove many instaces of frequent words (eg. "the")
    - generate training pairs and negative samples
    """

    def __init__(self, path, window_size=3, negative_size=5, max_tokens=None):
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
        self.negative_size = negative_size
        self.neg_probs = []
        self.max_tokens = max_tokens

    def raw_text_from_file(self):
        """
        This method loads raw text from file and stores it in self.raw_text.
        """
        with open(self.path, "r", encoding="utf8") as file:
            self.raw_text = file.read()

    def tokenization(self):
        """
        This method cleans and tokenize raw text.

        - remove HTML tags
        - convert to lowercase
        - remove non-alphabetic characters
        - removes to many spaces
        - split into words
        - selects first words in quantity of max_tokens

        output:
        -self.words : list of str
        """
        self.raw_text = re.sub(r"<[^>]+>", " ", self.raw_text)
        self.raw_text = self.raw_text.lower()
        self.raw_text = re.sub(r"[^a-z\s]", " ", self.raw_text)
        self.raw_text = re.sub(r"\s+", " ", self.raw_text)

        words = self.raw_text.split()

        if self.max_tokens is not None:
            words = words[:self.max_tokens]

        self.words = words

    def build_vocab(self):
        """
        This method builds vocabulary and maps words to indexes.

        - count words
        - create list vocabulary
        - word to index
        - index to word

        output:
        -self.vocab : list
        -self.vocab_size : int
        -self.word_to_index : dictionary
        -self.index_to_word : dictionary
            """
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
        """
        This method converts tokenized text into indexes.

        output
        -self.text_as_indexes : list of int
        """
        self.text_as_indexes = [self.word_to_index[word] for word in self.words]

    def generate_pairs(self):
        """
        This method generates all (center, context) pairs with size of self.window_size.

        return
        -np.ndarray of shape (N, 2)
        """
        pairs = []

        for i in range(len(self.text_as_indexes)):
            center_word = self.text_as_indexes[i]

            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i and 0 <= j < len(self.text_as_indexes):
                    context_word = self.text_as_indexes[j]
                    pairs.append((center_word, context_word))

        return np.array(pairs)

    def subsample(self, t=1e-5):
        """
        This method removes many instances of frequent wards.

        formula: P(discard) = 1 - sqrt(t / f)

         f - word frequency
        t - controls aggressiveness of subsampling

        effect
        -reduces dominance of frequent words (e.g. "the").
        """
        total_count = sum(self.word_counts.values())

        freqs = {w: c / total_count for w, c in self.word_counts.items()}

        new_words = []

        for w in self.words:
            f = freqs[w]

            p_discard = 1 - np.sqrt(t / f)

            if np.random.rand() > p_discard:
                new_words.append(w)

        self.words = new_words

    def generate_negative_words(self, positive):
        """
        This method generates negative samples for a given positive context word.

        returns
        -list of int: negative word indexes, excluding the positive word
        """
        negs = []
        while len(negs) < self.negative_size:
            w = np.random.choice(self.vocab_size)
            if w != positive: negs.append(w)
        return negs

    def prepere_data(self):
        """
        This method executes full preprocessing pipeline.

        Steps:
        1. Load raw text
        2. Tokenize
        3. Build vocabulary
        4. Apply subsampling
        5. Rebuild vocabulary
        6. Convert text to indexes
        """
        self.raw_text_from_file()
        self.tokenization()
        self.build_vocab()
        self.subsample()
        self.build_vocab()
        self.text_to_indexes()
