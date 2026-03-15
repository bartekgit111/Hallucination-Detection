import re

class data():
    def __init__(self,path):
        self.path = path
        self.raw_text = ""
        self.words = []

    def raw_text_from_file(self):
        with open(self.path, "r", encoding="utf8") as file:
            self.raw_text = file.read()

    def tokenization(self):
        self.raw_text = re.sub(r"<[^>]+>", " ", self.raw_text)
        self.raw_text = self.raw_text.lower()
        self.raw_text = re.sub(r"[^a-z\s]", " ", self.raw_text)
        self.raw_text = re.sub(r"\s+", " ", self.raw_text)

        self.words  = self.raw_text.split()

wiki = data("data/enwiki8")
wiki.raw_text_from_file()
wiki.tokenization()

print(wiki.words[:20])