# sys.argv[1:] defines Wikipedia page titles
# This example measures WMDs from the first page to all the rest

from collections import Counter
import sys

import numpy
import spacy
import requests
from wmd import WMD

# Load English tokenizer, tagger, parser, NER and word vectors
print("loading spaCy")
nlp = spacy.load("en")

# List of page names we will fetch from Wikipedia and query for similarity
titles = sys.argv[1:] or ["Germany", "Spain", "Google"]

documents = {}
for title in titles:
    print("fetching", title)
    pages = requests.get(
        "https://en.wikipedia.org/w/api.php?action=query&format=json&titles=%s"
        "&prop=extracts&explaintext" % title).json()["query"]["pages"]
    print("parsing", title)
    text = nlp(next(iter(pages.values()))["extract"])
    tokens = [t for t in text if t.is_alpha and not t.is_stop]
    words = Counter(t.text for t in tokens)
    orths = {t.text: t.orth for t in tokens}
    sorted_words = sorted(words)
    documents[title] = (title, [orths[t] for t in sorted_words],
                        numpy.array([words[t] for t in sorted_words],
                                    dtype=numpy.float32))


# Hook in WMD
class SpacyEmbeddings(object):
    def __getitem__(self, item):
        return nlp.vocab[item].vector

calc = WMD(SpacyEmbeddings(), documents)
print("calculating")
# Germany shall be closer to Spain than to Google
for title, relevance in calc.nearest_neighbors(titles[0]):
    print("%24s\t%s" % (title, relevance))
