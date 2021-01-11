import logging

from features_processing.bert_tokenizer import BERTTokenizer
from features_processing.keras_toknizer import KerasTokenizer
from features_processing.torch_tokenizer import TorchTokenizer


def get_processor(args):
    proc_type = args['type']
    parmas = args['parmas']
    logging.info("Pre-processing: {}".format(proc_type))
    logging.info("Pre-processing: {}".format(parmas))
    proc = None
    if proc_type =='keras_tokenizer': #
        proc = KerasTokenizer(**parmas)
    if proc_type =='bert_tokenizer': #
        proc = BERTTokenizer(**parmas)
    if proc_type =='torch_tokenizer': #
        proc = TorchTokenizer(**parmas)

    return proc