from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import logging
from keras.utils import pad_sequences

class KerasTokenizer():
    def __init__(self, mode, vocab_size , pad_length):
        self.mode = mode
        self.vocab_size = vocab_size
        self.pad_length = pad_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size)

    def fit(self, x):
        logging.info('fitting Keras Tokenizer')
        self.tokenizer.fit_on_texts(x)
        logging.info('document_count {}'.format(self.tokenizer.document_count))
        logging.info('num_words {}'.format(self.tokenizer.num_words))

    def transform(self, x):
        if self.mode =='seq':
            ret= self.tokenizer.texts_to_sequences(x)
            ret = pad_sequences(ret, maxlen=self.pad_length)
            print('seq keras tokenizer')
            # print (len(ret[0]))
            # print (ret[0])
            # print (ret.shape)
            return ret

        return self.tokenizer.texts_to_matrix(x, mode = self.mode)