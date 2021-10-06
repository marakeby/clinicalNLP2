from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import logging
from transformers import  AutoTokenizer

def truncate(x, n):
    ret = [x[0]] + x[-n:]
    return ret
        

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

#         ret = self.tokenizer(list(x), truncation=self.truncation, padding=self.padding, max_length= self.max_length)
        tokens = self.tokenizer.batch_encode_plus(list(x), max_length=self.max_length, padding='max_length', truncation=False)

        if self.truncation: # truncate the beginning of the text
        
            trucated_tokens={}
            for k in tokens.keys():
                print(k)
                trucated_tokens[k] = [truncate(l, self.max_length-1) for l in tokens[k]]

            ret = trucated_tokens
        else:
            ret = tokens  
            

        return ret
