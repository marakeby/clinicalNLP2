import logging
import os

import datetime
import numpy as np
# import theano
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


from model.model_utils import get_layers

# theano.config.openmp = False

class Model(BaseEstimator):
    def __init__(self, build_fn, **sk_params):
        params = sk_params
        params['build_fn'] = build_fn
        self.set_params(params)

    def set_params(self, sk_params):
        self.params = sk_params
        # print sk_params
        self.build_fn = sk_params['build_fn']
        self.sk_params = sk_params
        self.batch_size = sk_params['fitting_params']['batch_size']
        self.model_params = sk_params['model_params']
        self.nb_epoch = sk_params['fitting_params']['epoch']
        self.verbose = sk_params['fitting_params']['verbose']
        # self.verbose = sk_params['fitting_params']['verbose']
        self.select_best_model = sk_params['fitting_params']['select_best_model']

        if 'x_to_list' in sk_params['fitting_params']:
            self.x_to_list = sk_params['fitting_params']['x_to_list']
        else:
            self.x_to_list = False

        if 'feature_importance' in sk_params:
            self.feature_importance = sk_params['feature_importance']

        if 'loss' in sk_params['model_params']:
            self.loss = sk_params['model_params']['loss']
        else:
            self.loss = 'binary_crossentropy'
        if 'reduce_lr' in sk_params['fitting_params']:
            self.reduce_lr = sk_params['fitting_params']['reduce_lr']
        else:
            self.reduce_lr = False
        if 'reduce_lr_after_nepochs' in sk_params['fitting_params']:
            self.reduce_lr_after_nepochs = sk_params['fitting_params']['reduce_lr_after_nepochs']
        else:
            self.reduce_lr_after_nepochs = False

        pid = os.getpid()
        timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}-{0:%S}'.format(datetime.datetime.now())
        self.save_filename = './temp/' + sk_params['fitting_params']['save_name'] + str(pid) + timeStamp
        self.shuffle = sk_params['fitting_params']['shuffle']
        self.monitor = sk_params['fitting_params']['monitor']
        self.duplicate_samples = False
        if 'duplicate_samples' in sk_params:
            self.duplicate_samples = sk_params['duplicate_samples']

        if 'n_outputs' in sk_params['fitting_params']:
            self.n_outputs = sk_params['fitting_params']['n_outputs']
        else:
            self.n_outputs = 1

        # print 'self.n_outputs {}'.format(self.n_outputs)

    def get_params(self, deep=False):

        return self.params

    def fit(self, X_train, y_train, X_val = None, y_val =None):

        self.model = self.build_fn(**self.model_params)

        callbacks  = []

        if self.reduce_lr:
            reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=0.5,
                                          patience=2, min_lr=0.000001, verbose=2, mode='auto')
            logging.info("adding a reduce lr on Plateau callback%s "%reduce_lr)
            callbacks.append(reduce_lr)

        if self.select_best_model:
            saving_callback = ModelCheckpoint(self.save_filename, monitor=self.monitor, verbose=1, save_best_only=True,
                                              mode='auto')
            logging.info("adding a saving_callback%s " % saving_callback)
            callbacks.append(saving_callback)

        if self.reduce_lr_after_nepochs:
            reduce_lr = decay_lr(5, 0.5)
            logging.info("adding a reduce lr callback %s " % reduce_lr)
            callbacks.append(reduce_lr)

        if self.n_outputs > 1:
            y_train = [y_train]* self.n_outputs
            y_val = [y_val]* self.n_outputs

        if not X_val is None:
            validation_data = [X_val, y_val]
        else:
            validation_data = []

        logging.info(self.model.summary())
        history = self.model.fit(X_train, y_train, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose, validation_data=validation_data, callbacks=callbacks, shuffle=self.shuffle)


        weights= self.model.layers[1].get_weights()
        # print 'weights', len(weights)


        if hasattr(self, 'feature_importance'):
            if self.feature_importance=='gradient':
                self.coef_ = get_gradient_weights(self.model, X_train, y_train)
            elif self.feature_importance=='gradient_with_repeated_outputs':
                self.coef_ = get_gradient_weights_with_repeated_output(self.model, X_train, y_train)
            elif self.feature_importance == 'permutation':
                self.coef_ = get_permutation_weights(self.model, X_train, y_train)
            elif self.feature_importance == 'linear':
                self.coef_ = get_weights_linear_model(self.model, X_train, y_train)
            elif self.feature_importance == 'one_to_one':
                weights = self.model.layers[1].get_weights()
                switch_layer_weights = weights[0]
                self.coef_ = np.abs(switch_layer_weights)
            else:
                self.coef_ = None

        # get_deconstruction_weights(self.model)
        return self

    def predict(self, X_test):
        if self.select_best_model:
            logging.info("loading model %s"%self.save_filename)
            self.model.load_weights(self.save_filename)

        prediction_scores = self.model.predict(X_test)
        if type(prediction_scores) ==list:
            prediction_scores= prediction_scores[-1]

        if self.loss == 'hinge':
            prediction = np.where(prediction_scores >= 0.0, 1., 0.)
        else:
            prediction = np.where(prediction_scores >= 0.5, 1., 0.)

        return prediction

    def predict_proba(self, X_test):
        prediction_scores = self.model.predict(X_test)
        if type(prediction_scores) ==list:
            prediction_scores= prediction_scores[0]

        if type(X_test) is list:
            n_samples = X_test[0].shape[0]
        else:
            n_samples = X_test.shape[0]
        ret = np.ones((n_samples,2))
        ret[:,0]= 1.-prediction_scores.ravel()
        ret[:,1] =prediction_scores.ravel()
        return ret

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return accuracy_score(y_test, y_pred)

    def get_layer_outputs(self, X):
        inp = self.model.input
        layers = get_layers(self.model)
        outputs = [layer.output for layer in layers]  # all layer outputs
        functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        layer_outs = [func([X, 1.]) for func in functors]

        return layer_outs

    def save_model(self, filename):
        # model_json = self.model.to_json()
        # json_file_name = filename.replace('.h5', '.json')
        # with open(json_file_name, "w") as json_file:
        #     json_file.write(model_json)
        # load_weights()
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model = self.build_fn(**self.model_params)
        self.model.load_weights(filename)
        return self.model

