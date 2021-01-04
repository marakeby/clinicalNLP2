# import cPickle
import _pickle as cPickle
import logging

import time

from keras.models import Sequential


def save_model(model, filename):
    print ('saving model in', filename)
    f = file(filename + '.pkl', 'wb')
    import sys
    sys.setrecursionlimit(100000)
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_model(file_name):
    f = file(file_name + '.pkl', 'rb')
    # theano.config.reoptimize_unpickled_function = False
    start = time.time()
    model = cPickle.load(f)
    end = time.time()
    elapsed_time = end - start
    return model

def print_model(model, level=1):
    for i, l in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(l) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))
            print_model(l, level+1)
        else:
            logging.info( '{} {} {} {}'.format(indent , i, l.name , l.output_shape))

def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level+1))
        else:
            layers.append(l)

    return layers




def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output