__author__ = 'marakeby'

from sklearn import preprocessing as p
from sklearn.decomposition import PCA
import numpy as np
import logging

from preprocessing.clean_text import CleanText


def get_processor( args):
    print (args)
    proc_type = args['type']
    logging.info("Pre-processing: %s", proc_type)
    # params = args['params']
    if proc_type =='standard': # 0 mean , 1 variance
        if 'params' in args:
            p1 = args['params']
            proc =  p.StandardScaler(**p1)
        else:
            proc =  p.StandardScaler()
    elif proc_type =='normalize': #  1 norm
        proc = p.Normalizer()

    # elif proc_type =='abs': #  1 norm
    #     proc = np.abs
    elif proc_type == 'scale': # 0:1 scale
        if 'params' in args:
            p1 = args['params']
            proc = p.MinMaxScaler(**p1)
        else:
            proc = p.MinMaxScaler()

    elif proc_type =='log': # to be implemented
        proc = None #TODO: implement log scaling
    elif proc_type == 'tissue-specific':
        # from tissue_specefic import tissue_specific
        proc = tissue_specific()
    elif proc_type == 'smart':
        p1 = args['params']
        proc = get_processor(p1)
        proc = SmartPreprocesor(proc)
    elif proc_type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfTransformer

        p1 = args['params']
        proc = TfidfTransformer(**p1)

    elif proc_type=='clean_text':
        proc = CleanText()
    else:
        proc = None

    return proc


def remove_outliers(y):
    m = np.mean(y)
    s = np.std(y)

    import copy
    y2 = copy.deepcopy(y)
    n = 4
    y2[y>m+ n*s] = m+n*s
    y2[y<m -n*s] = m-n*s
    return y2
