import logging

from data.manual_labels.data_reader import ManualLabels
# from data.medical_notes.data_reader import ClinicalData
import pandas as pd
import numpy as np
class Data():
    def __init__(self, type, id, params):

        self.data_type = type
        self.data_params = params

        if self.data_type == 'survival':
            self.data_reader = ClinicalData(**params)
        if self.data_type == 'manual_label':
            self.data_reader = ManualLabels(**params)
        else:
            logging.error('unsupported data type')
            raise ValueError('unsupported data type')


    def get_train_validate_test(self):
        # x= self.data_reader.x
        # y = self.data_reader.y
        # y = self.data_reader.y
        # columns = self.data_reader.columns
        # info = pd.DataFrame(index=self.data_reader.info)
        # logging.info('diving data into training  and test')
        #
        # x_train, x_test, y_train, y_test, info_train, info_test= train_test_split(x, y,info, test_size=0.30, stratify=y,
        #                                                     random_state=422342)
        #
        # logging.info('x_train {} \nx_test {} \ny_train {}\ny_test {}' .format(
        #     x_train.shape, x_test.shape, y_train.shape, y_test.shape))
        # return x_train, x_test, y_train, y_test, info_train.copy(), info_test.copy(), columns
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns =  self.data_reader.get_train_validate_test()
        #combine training and validation datasets
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = pd.concat([info_train,info_validate ])

        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    def get_relevant_features(self):
        if hasattr(self.data_reader, 'relevant_features'):
            return self.data_reader.get_relevant_features()
        else:
            return None


