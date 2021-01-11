from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import logging
from transformers import BertTokenizer, AutoTokenizer


class BERTTokenizer():
    def __init__(self, model_name, truncation, padding, max_length=None):
        self.model_name = model_name
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self, x):
        logging.info('fitting BERT Tokenizer')
        pass

    def transform(self, x):

        ret = self.tokenizer(list(x), truncation=self.truncation, padding=self.padding, max_length= self.max_length)
        print('type ret',type(ret))
        # print(ret[100])
        print(len(ret))

        return ret
        # return self.tokenizer(list(x), truncation=self.truncation, padding=self.padding)
