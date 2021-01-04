import logging
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.regularizers import l1, l2
from model.model_utils import apply_models, print_model
import os
import numpy as np


def build_simple(vocab_size, embedding_dims, input_maxlen, hidden_dims, kernel_size, filters, dropout, strides, optimizer, activation, w_reg):
    model = Sequential()

    model.add(Embedding(vocab_size,
                        embedding_dims,
                        input_length=input_maxlen))
    model.add(Dropout(dropout))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation=activation,
                     strides=strides))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation(activation))

    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation(activation))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def build_glove(vocab_size, embedding_dims, input_maxlen, hidden_dims, kernel_size, filters, dropout, strides, optimizer, activation, w_reg):


    BASE_DIR = ''
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, embedding_dims))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    model = Sequential()

    model.add(Embedding(vocab_size,
                        embedding_dims,
                        weights=[embedding_matrix],
                        input_length=input_maxlen))
    model.add(Dropout(dropout))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation=activation,
                     strides=strides))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation(activation))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def build_full_network(optimizer, n_features, n_hids, dropouts, w_reg, activation='tanh', reg_type='l1'):

        logging.info('Compiling...')
        ins = Input(shape=(n_features,), dtype='float32', name='inputs')
        layers = []

        for i, n in enumerate(n_hids):
            if reg_type == 'l1':
                wreg = l1(w_reg[i])
            elif reg_type == 'l2':
                wreg = l2(w_reg[i])

            if i == 0:
                layer = Dense(n, input_shape=(n_features,), activation=activation, W_regularizer=wreg,
                              name='h' + str(i))
            else:
                layer = Dense(n, activation=activation, W_regularizer=l1(w_reg[i]), name='h' + str(i))

            layers.append(layer)
            drop = dropouts[i]
            layers.append(Dropout(drop))

        merged = apply_models(layers, ins)
        output_layer = Dense(1, activation='sigmoid')
        outs = output_layer(merged)

        model = Model(input=[ins], output=[outs])

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        logging.info('done compiling')

        print_model(model)
        logging.info(model.summary())
        logging.info('# of trainable params of the model is %s' % model.count_params())
        return model