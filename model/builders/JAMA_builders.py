
import logging
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, concatenate
from tensorflow.keras.regularizers import l1, l2
from model.model_utils import apply_models, print_model
import os
import numpy as np


# vocab_size = 20000
# input_maxlen = 1000
# batch_size = 32
# embedding_dims = 300
# filters = 300
# kernel_size = 10
# hidden_dims = 250
# epochs = 2


def any_cancer_JAMA(vocab_size, embedding_dims, input_maxlen, hidden_dims, kernel_size, filters, dropout=0.2, strides=1,
                    optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dims,
                        input_length=input_maxlen))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation=activation,
                     strides=strides))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add vanilla hidden layers:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))

    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    logging.info('number of trainable parmas {}'.format(model.count_params()))
    logging.info(model.summary())
    return model


def model_JAMA(vocab_size, embedding_dims, input_maxlen, hidden_dims, kernel_size, filters, dropout=0.2, strides=1,
                    optimizer='adam', activation='relu'):
    print('Build model...')
    main_input = Input(shape=(input_maxlen,), dtype='int32')

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    embed_layer = Embedding(vocab_size, embedding_dims, input_length=input_maxlen)(main_input)
    x = embed_layer

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    x = Conv1D(filters,
               kernel_size,
               padding='valid',
               activation=activation,
               strides=strides)(x)

    # we use max pooling:
    x = (GlobalMaxPooling1D())(x)

    # bring in any cancer model
    # We add a vanilla hidden layer:
    x = Dense(hidden_dims)(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x1 = Dense(hidden_dims)(x)
    x1 = Dropout(dropout)(x1)
    anycancer_output = Dense(1, activation="sigmoid", name="anycancer_output")(x1)

    # the hidden layer for prediction of any cancer is concatenated to the final common hidden layer
    x = concatenate([x, x1])
    x = Dense(hidden_dims)(x)
    x = Dropout(dropout)(x)
    x = Activation(activation)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    main_output = Dense(1, activation="sigmoid", name="main_output")(x)

    model = Model(inputs=main_input, outputs=[main_output, anycancer_output])

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
