import re
from copy import copy

import numpy as np
import logging

from sklearn.base import BaseEstimator


def preprocess_text(sen, remove_tag=True, remove_special_chars=True, remove_numbers=False, remove_single_chars=True, remove_multiple_sapces=True):
    sentence =copy(sen)
    # Removing html tags
    if remove_tag:
        sentence = remove_tags(sen)

    # Remove punctuations and numbers
    if remove_special_chars:
        sentence = re.sub('[^0-9a-zA-Z:,]', ' ', sentence)

    if remove_numbers:
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    if remove_single_chars:
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    if remove_multiple_sapces:
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

class CleanText(BaseEstimator):
    def __init__(self, lower_case =False):
        logging.info('preprocessing: cleaning text')
        self.lower_case= lower_case


    def fit(self, X):
        pass

    def transform(self,X, y='deprecated', copy=None):
        ret_list = []
        sentences = list(X)
        for sen in sentences:
            if self.lower_case:
                sen = sen.lower()
            ret_list.append(preprocess_text(sen))
        return np.array(ret_list)




