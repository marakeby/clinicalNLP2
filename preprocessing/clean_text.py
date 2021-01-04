import re
import numpy as np
import logging

from sklearn.base import BaseEstimator


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('[^0-9a-zA-Z:,]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

class CleanText(BaseEstimator):
    def __init__(self):
        logging.info('preprocessing: cleaning text')


    def fit(self, X):
        pass

    def transform(self,X, y='deprecated', copy=None):
        ret_list = []
        sentences = list(X)
        for sen in sentences:
            ret_list.append(preprocess_text(sen))
        return np.array(ret_list)




