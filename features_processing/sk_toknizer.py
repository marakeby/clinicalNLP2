import logging
from sklearn.feature_extraction.text import TfidfVectorizer


class SKTokenizer():
    def __init__(self):
        self.tokenizer = TfidfVectorizer()

    def fit(self, x):
        logging.info('fitting Scikit-Learn Tokenizer')
        self.tokenizer.fit(x)

    def transform(self, x):
        return self.tokenizer.transform(x).toarray()
