import logging

from features_processing.bert_tokenizer import BERTTokenizer

from features_processing.sk_toknizer import SKTokenizer
from features_processing.torch_tokenizer import TorchTokenizer


def get_processor(args):
    proc_type = args['type']
    parmas = args['params']
    logging.info("Pre-processing: {}".format(proc_type))
    logging.info("Pre-processing: {}".format(parmas))
    proc = None
    if proc_type =='keras_tokenizer': #
        from features_processing.keras_toknizer import KerasTokenizer
        proc = KerasTokenizer(**parmas)
    if proc_type =='bert_tokenizer': #
        proc = BERTTokenizer(**parmas)
    if proc_type =='torch_tokenizer': #
        proc = TorchTokenizer(**parmas)
    if proc_type =='sk_tokenizer':
        proc = SKTokenizer()

    return proc