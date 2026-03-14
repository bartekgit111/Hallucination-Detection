# with open("data/enwik8", "r", encoding="utf8") as f:
#     text = f.read()
#
# print(len(text))
# print(text[:500])

import re

with open("data/enwik8", "r", encoding="utf8") as f:
    text = f.read()

# usuń tagi XML
text = re.sub(r"<[^>]+>", " ", text)

# zamień na małe litery
text = text.lower()

# usuń znaki specjalne
text = re.sub(r"[^a-z\s]", " ", text)

# usuń podwójne spacje
text = re.sub(r"\s+", " ", text)

words = text.split()

print("number of words:", len(words))
print(words[:50])
