import logging
import datetime

import yaml
from sklearn.model_selection import StratifiedKFold, KFold
from data.data_access import Data
# from features_processing.feature_scaler import FeatureScaler
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
import pandas as pd
from os.path import join, exists

from utils.plots import plot_box_plot_groupby
from utils.rnd import set_random_seeds

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

def get_mean_variance(scores):
    df = pd.DataFrame(scores)
    return df, df.mean(), df.std()

class CrossvalidationPipeline(OneSplitPipeline):
    def __init__(self, data_params, pre_params, feature_params,model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, data_params, pre_params,feature_params, model_params, pipeline_params, exp_name)

    def run(self, n_splits =5):
        # logging
        logging.info( 'loading data....')
        test_scores_df = pd.DataFrame()
        model_names = []
        model_list = []
        for data_params in self.data_params:
            print ('data_params',data_params)
            data_id =  data_params['id']
            data = Data(**data_params)
            # data = Data(**self.data_params)

            # get data
            # X, y, info, cols = data.get_data()
            x_train, x_test, y_train, y_test, info, info_test, cols = data.get_train_test()
            X = x_train
            y = y_train

            # get model
            logging.info('fitting model ...')
            list_model_scores= []
            model_names = []
            if type(self.model_params) == list:
                for m in self.model_params:
                    # model_name = m['type']
                    if 'id' in m:
                        model_name = m['id']
                    else:
                        model_name = m['type']


                    set_random_seeds(random_seed=818)


                    logging.info('fitting model ...')

                    model_name_extended = '{}_{}'.format(data_id, model_name)
                    scores = self.train_predict_crossvalidation( m, X, y, info, cols, model_name_extended)
                    scores_df, scores_mean, scores_std = get_mean_variance(scores)
                    list_model_scores.append(scores_df)
                    model_names.append(model_name)
                    
                    
                    self.save_score(scores_df, scores_mean, scores_std, model_name_extended)
                    scores_df['model'] = model_name
                    scores_df['data'] = data_id
                    test_scores_df = test_scores_df.append(scores_df)
                    # test_scores.append(scores_df)
                    logging.info('scores')
                    logging.info(scores_df)
                    logging.info('mean')
                    logging.info( scores_mean)
                    logging.info('std')
                    logging.info(scores_std)
                    # logging.info(f'{model_name} Training Time: {mins}m {secs}s')

            df = pd.concat(list_model_scores, axis=1, keys=model_names)
            df.to_csv(join(self.directory,'folds_{}.csv'.format(data_id)))
            # print(df.head())
            # plot_box_plot(df, self.directory)

        # test_scores_df= pd.concat(test_scores, axis=1)
        test_scores_df.to_csv(join(self.directory,'folds.csv'))
        plot_box_plot_groupby(test_scores_df, self.directory, groupby=['model', 'data'])
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(figsize=(10,8))
        # plt.suptitle('')
        # test_scores_df.boxplot(column=['accuracy'], by=['model', 'data'], ax=ax)
        # plt.savefig(join(self.directory,'accuracy.png'))
            # test_scores.append(test_score)
            # model_names.append(model_name)
            # logging.info(f'{data_id} Training Time: {mins}m {secs}s')

            

        return scores_df

    def save_prediction(self, info, y_pred, y_pred_score,  y_test, fold_num, model_name,  training=False):
        if training:
            file_name = join(self.directory , model_name+ '_traing_fold_' + str(fold_num) + '.csv')
        else:
            file_name = join(self.directory , model_name+ '_testing_fold_' + str(fold_num) + '.csv')
        logging.info("saving : %s" % file_name)
        info['pred'] = y_pred
        info['pred_score'] = y_pred_score
        info['y'] = y_test
        info.to_csv(file_name)

    # class PateintKFold():
    #     def __init__(self,info, n_splits=5, random_state=123, shuffle=True):
    #         self.n_splits = n_splits
    #         self.random_state = random_state
    #         self.shuffle = shuffle
    #         self.skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
    #         self.pateints = info['DFCI_MRN'].unique()
    #     def split(self, X, y, groups=None):
    #         self.skf.split(self.pateints)

    def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name):
        model = get_model(model_params)
        n_splits = self.pipeline_params['params']['n_splits']
        kf = KFold(n_splits=n_splits, random_state=123, shuffle=True)
        i=0
        scores = []
        # for train_index, test_index in skf.split(X, y.ravel()):
        pateints = info['DFCI_MRN'].unique()
        for train_index, test_index in kf.split(pateints):
            logging.info('fold # ----------------%d---------'%i)
            print ('train patients', train_index.shape)
            print ('test patients', test_index.shape)
            train_patients = pateints[train_index]
            test_patients = pateints[test_index]
            # logging.info(train_patients)
            # logging.info(test_patients)
            train_index = info['DFCI_MRN'].isin(train_patients)
            test_index = info['DFCI_MRN'].isin(test_patients)

            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = info[train_index]
            info_test = info[test_index]
            # info_train = pd.DataFrame(index = info.iloc[train_index])
            # info_test = pd.DataFrame(index = info.iloc[info_test])
            # info_test = info.iloc[test_index,:].copy()
            x_train, x_test = self.preprocess(x_train, x_test)

            logging.info('x_trian {}   x_test{} '.format(x_train.shape,  x_test.shape))
            logging.info('y_trian {}   y_test{} '.format(y_train.shape,  y_test.shape))


            # feature extraction
            logging.info('feature extraction....')
            x_train, x_test, x_test = self.extract_features(x_train, x_test, x_test)


            if 'fitting_params' in model_params['params']:
                if 'x_to_list' in model_params['params']['fitting_params']:
                    if model_params['params']['fitting_params']['x_to_list']:
                        x_train = self.get_list(x_train, cols)
                        x_test = self.get_list(x_test, cols)

            model = model.fit(x_train, y_train)
            y_pred_test, y_pred_test_scores, score_test, confusion_mtrx = self.predict(model, x_test, y_test)
            self.save_prediction(info_test, y_pred_test,y_pred_test_scores, y_test, i, model_name)

            if self.save_train:
                y_pred_train, y_pred_train_scores, score_train, confusion_mtrx = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name, training=True)

            scores.append(score_test)
            i += 1
        return scores

    def save_score(self, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory , model_name + '_params'  + '.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data':self.data_params, 'models':self.model_params, 'pre':self.pre_params,'pipeline': self.pipeline_params, 'scores':scores.to_json(), 'scores_mean':scores_mean.to_json(),'scores_std':scores_std.to_json() }, default_flow_style=False))




