from sklearn.metrics import confusion_matrix
from sklearn.utils.extmath import density
import numpy as np
from features_processing import feature_extraction
# from features_processing.feature_scaler import FeatureScaler
from utils.evaluate import evalualte
from os.path import join, exists
from model.model_factory import get_model
from os import makedirs

import logging
import datetime
from data.data_access import Data

from preprocessing import pre
import scipy.sparse
import yaml
import pandas as pd
from matplotlib import  pyplot as plt

from utils.plots import generate_plots, plot_roc, plot_confusion_matrix, plot_prc

# timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
from utils.rnd import set_random_seeds


class OneSplitPipeline:
    def __init__(self, data_params, pre_params, feature_params,  model_params, pipeline_params, exp_name):


        self.data_params = data_params
        self.pre_params = pre_params
        self.features_params = feature_params
        self.model_params = model_params
        self.exp_name = exp_name
        self.pipeline_params = pipeline_params
        if 'save_train' in pipeline_params['params']:
            self.save_train =  pipeline_params['params']['save_train']
        else:
            self.save_train = False

        if 'eval_dataset' in pipeline_params['params']:
            self.eval_dataset =  pipeline_params['params']['eval_dataset']
        else:
            self.eval_dataset = 'validation'

        self.prapre_saving_dir()

    def prapre_saving_dir(self):
        # self.directory = self.exp_name +  timeStamp
        self.directory = self.exp_name
        if not exists(self.directory):
            makedirs(self.directory)

    def save_prediction(self,info, y_pred,y_pred_score, y_test, model_name,  training= False):

        if training:
            file_name = join(self.directory ,  model_name+ '_traing.csv')
        else:
            file_name = join(self.directory ,  model_name+ '_testing.csv')
        logging.info("saving : %s" % file_name)

        info = info.copy()
        info['pred'] = y_pred
        info['pred_score'] = y_pred_score
        info['y'] = y_test
        info.to_csv(file_name)

    def get_list(self,x, cols):
        x_df = pd.DataFrame(x, columns=cols)

        genes = cols.get_level_values(0).unique()
        genes_list = []
        input_shapes = []
        for g in genes:
            g_df = x_df.loc[:, g].as_matrix()
            input_shapes.append(g_df.shape[1])
            genes_list.append(g_df)
        return genes_list

    def run(self):
        # logging
        logging.info( 'loading data....')


        test_scores = []
        model_names = []
        model_list = []
        cnf_matrix_list = []
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        auc_fig = plt.figure()
        prc_fig = plt.figure()

        for data_params in self.data_params:
            print ('data_params',self.data_params)
            data_id =  data_params['id']
            data = Data(**data_params)
            # get data
            x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()

            print( 'info', type(info_train), type(info_validate), info_test.shape)
            logging.info( 'info {} {} {}'.format(info_train.shape, info_validate.shape, info_test.shape))
            logging.info('x {} {} {}'.format( x_train.shape, x_validate.shape, x_test.shape))
            logging.info('y {} {} {}'.format( y_train.shape, y_validate.shape, y_test.shape))

            # pre-processing
            logging.info('preprocessing....')
            x_train, x_test = self.preprocess(x_train, x_test)

            logging.info('feature extraction....')
            x_train, x_validate, x_test = self.extract_features(x_train, x_validate, x_test)


            for m in self.model_params:
                # get model
                set_random_seeds(random_seed=20080808)

                model = get_model(m)
                                
                if m['type'] == 'nn':
                    model = model.fit(x_train, y_train, x_validate, y_validate)
                else:
                    model = model.fit(x_train, y_train)

                logging.info('predicting')
                if self.eval_dataset == 'validation':
                    x_t = x_validate
                    y_t = y_validate
                    info_t = info_validate
                else:
                    x_t = x_test
                    y_t = y_test
                    info_t= info_test

                y_pred_test, y_pred_test_scores, test_score, cnf_matrix = self.predict(model, x_t, y_t)

                cnf_matrix_list.append(cnf_matrix)
                if 'id' in m:
                    model_name = m['id']
                else:
                    model_name = m['type']

                model_name = model_name+'_'+ data_id
                test_scores.append(test_score)
                model_names.append(model_name)

                logging.info('saving results')
                self.save_score(test_score, model_name)
                self.save_prediction(info_t, y_pred_test,y_pred_test_scores, y_t, model_name)
                plot_roc(auc_fig, y_t, y_pred_test_scores, self.directory, label=model_name)
                plot_prc(prc_fig, y_t, y_pred_test_scores, self.directory, label=model_name)
                # plt.savefig(join(self.directory, 'auprc_curves'))

                if self.save_train:
                    y_pred_train, y_pred_train_scores, score, train_cnf_matrix= self.predict(model, x_train, y_train)
                    self.save_prediction(info_train, y_pred_train,y_pred_train_scores,  y_train, model_name,  training=True)

        classes = np.unique(y_train)
        auc_fig.savefig(join(self.directory, 'auc_curves'))
        prc_fig.savefig(join(self.directory, 'auprc_curves'))

        test_scores = pd.DataFrame(test_scores, index=model_names)
        generate_plots(test_scores, self.directory)
        self.save_all_scores(test_scores)
        # self.plot_coef(model_list)
        self.save_cnf_matrix(cnf_matrix_list, model_names, classes)

        return test_scores

    def save_cnf_matrix(self, cnf_matrix_list, model_list, classes):
        for cnf_matrix, model in zip(cnf_matrix_list, model_list):
            plt.figure()
            # plot_confusion_matrix(cnf_matrix, classes=['Primary','Metastatic'],
            plot_confusion_matrix(cnf_matrix, classes=classes,
                                  title='Confusion matrix, without normalization')
            file_name = join(self.directory, 'confusion_' + model)
            plt.savefig(file_name)

            plt.figure()
            plot_confusion_matrix(cnf_matrix, normalize=True, classes=classes,
                                  title='Normalized confusion matrix')
            file_name = join(self.directory, 'confusion_normalized_' + model)
            plt.savefig(file_name)

            # Plot normalized confusion matrix
            # plt.figure()
            # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
            #                       title='Normalized confusion matrix')

    def plot_coef(self, model_list):
        for model, model_name in model_list:
            plt.figure()
            file_name = join(self.directory, 'coef_'+model_name)
            for coef in model.coef_:
                plt.hist(coef, bins=20)
            # plt.hist(model.coef_[1], bins=20)
            plt.savefig(file_name)

    def save_all_scores(self, scores):
        file_name = join(self.directory , 'all_scores.csv')
        scores.to_csv(file_name)

    def save_score(self, score, model_name):
        file_name = join(self.directory , model_name+ '_params.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump([self.data_params, self.model_params, self.pre_params, str(score)], default_flow_style=False))


    def predict(self, model, x_test, y_test ):
        logging.info('predicitng ...')
        y_pred_test = model.predict(x_test)
        if hasattr(model, 'predict_proba'):
            y_pred_test_scores =  model.predict_proba(x_test)[:,1]
        else:
            y_pred_test_scores = y_pred_test

        # y_pred_test_scores =  model.predict_proba(x_test)[:,1]
        logging.info('scoring ...')
        score = evalualte(y_test, y_pred_test, y_pred_test_scores)
        cnf_matrix = confusion_matrix(y_test, y_pred_test)
        return y_pred_test, y_pred_test_scores, score, cnf_matrix

    def preprocess(self, x_train, x_test):
        logging.info('preprocessing....')
        print(self.pre_params)
        proc = pre.get_processor(self.pre_params)
        if proc:
            proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            # if scipy.sparse.issparse(x_train):
            #     x_train = x_train.todense()
            #     x_test = x_test.todense()
        return x_train, x_test

    def extract_features(self, x_train, x_validate, x_test):
        if self.features_params== {}:
            return x_train, x_test
        logging.info('feature extraction ....')
        print (self.features_params)
        proc = feature_extraction.get_processor(self.features_params)
        if proc:
            proc.fit(x_train)
            logging.info('x_train')
            x_train = proc.transform(x_train)
            logging.info('x_test')
            x_test = proc.transform(x_test)
            logging.info('x_validate')
            x_validate = proc.transform(x_validate)

            # if scipy.sparse.issparse(x_train):
            #     x_train = x_train.todense()
            #     x_test = x_test.todense()
        return x_train,x_validate,  x_test