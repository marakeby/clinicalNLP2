# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
import logging
from collections import Counter
from torchtext.data.utils import get_tokenizer
import torch
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

class TorchTokenizer():

    def __init__(self, mode, vocab_size , pad_length):
        # self.tokenizer = get_tokenizer('spacy', language='en')
        self.tokenizer = get_tokenizer('basic_english')
        self.mode = mode
        self.vocab_size = vocab_size
        self.pad_length = pad_length
        # self.tokenizer = Tokenizer(num_words=self.vocab_size)

    def fit(self, x):
        logging.info('fitting Torch Tokenizer')
        self.vocab = build_vocab(x, self.tokenizer, self.vocab_size)

        # logging.info('document_count {}'.format(self.tokenizer.document_count))
        # logging.info('num_words {}'.format(self.tokenizer.num_words))

    def transform(self, x):
        logging.info('transform Torch Tokenizer')
        data = []
        PAD_IDX = self.vocab['<pad>']
        BOS_IDX = self.vocab['<bos>']
        EOS_IDX = self.vocab['<eos>']

        for string_ in list(x):
            tokenized = self.tokenizer(str(string_))
            ##  truncating
            if len(tokenized)>self.pad_length:
                tokenized = tokenized[-self.pad_length+2:]

            en_tensor_ = torch.tensor([self.vocab[token] for token in tokenized], dtype=torch.long)
            # data.append(en_tensor_)
            data.append(torch.cat([torch.tensor([BOS_IDX]), en_tensor_, torch.tensor([EOS_IDX])], dim=0))

            # data.append( en_tensor_)

        ## padding
        data_padded = pad_sequence(data, batch_first=True , padding_value=PAD_IDX)

        return data_padded

def build_vocab(x, tokenizer, max_vocab):
    counter = Counter()
    for string_ in x:
      counter.update(tokenizer(str(string_)))

    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], max_size=max_vocab)