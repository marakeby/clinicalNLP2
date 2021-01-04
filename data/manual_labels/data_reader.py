
import pandas as pd
from os.path import dirname, join
import numpy as np

from config_path import DATA_PATH

dir_path = dirname(dirname(__file__))
import logging

# processed_dir = join(dir_path, 'manual_labels/processed/')
# input_dir = join(dir_path, 'medical_notes/input/derived_data')
# input_dir = join(dir_path, 'medical_notes/input/radiology_impressions_derived_data')
input_dir = join(DATA_PATH, 'manual_labels/input')
processed_dir = join(dir_path, 'manual_labels/processed')
class ManualLabels():
    def __init__(self, outcome = 'any_cancer', text = 'NARR+IMPRESS', training_split=0):

        #which dataset to use for training {0: original training, 1, 2, 3..: decreasing sizes of training splits}

        self.training_split = training_split
        filename= join(input_dir, 'manual_label_analysis.csv')
        label_analysis = pd.read_csv(filename)

        # combine NARR_TXT and IMPRESS_TXT and get rid of carriage returns

        # label_analysis = label_analysis.assign(imaging_text=label_analysis.NARR_TXT + ' ' + label_analysis.IMPRESS_TXT)
        if text=='NARR+IMPRESS':
            label_analysis = label_analysis.assign(imaging_text=label_analysis.NARR_TXT + ' ' + label_analysis.IMPRESS_TXT)
        elif text=='IMPRESS':
            label_analysis = label_analysis.assign(imaging_text=label_analysis.IMPRESS_TXT)
        elif text == 'NARR':
            label_analysis = label_analysis.assign(imaging_text=label_analysis.NARR_TXT)

        label_analysis['imaging_text'] = label_analysis.imaging_text.str.replace(r'\r\n', ' ')
        label_analysis['imaging_text'] = label_analysis.imaging_text.str.replace(r'\r', ' ')
        label_analysis['imaging_text'] = label_analysis.imaging_text.str.replace(r'\n', ' ')

        # drop duplicate reports
        label_analysis = label_analysis.drop_duplicates(subset='imaging_text')

        # drop outside scans
        label_analysis = label_analysis.query('imaging_text.str.contains("it has been imported") == False')
        # logging.info(label_analysis.info())

        x = label_analysis
        x = x.assign(response=np.where(x.redcap_resp_prog == 1, 1, 0))
        x = x.assign(progression=np.where(x.redcap_resp_prog == 4, 1, 0))

        info = x[['DFCI_MRN', 'ehr_scan_date', 'PROC_DESCR']]

        if outcome =='any_cancer':
            y = x['any_cancer'].copy()
        elif outcome =='response':
            y = x['response'].copy()
        elif outcome =='progression':
            y = x['progression'].copy()

        self.x = x.imaging_text.values
        self.y = y.values
        self.info = info
        self.columns = []

        print (x.shape, y.shape)

    def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns
        # get training and validation patients
        training_file= 'training_mrns_{}.csv'.format(self.training_split)
        training_mrns = pd.read_csv(join(processed_dir, training_file))

        validation_mrns = pd.read_csv(join(input_dir, 'validation_mrns.csv'))
        testing_mrns = pd.read_csv(join(input_dir, 'truetest_mrns.csv'))

        ind_train = info['DFCI_MRN'].isin(training_mrns.DFCI_MRN)
        ind_validate = info['DFCI_MRN'].isin(validation_mrns.DFCI_MRN)
        ind_test = info['DFCI_MRN'].isin(testing_mrns.DFCI_MRN)

        x_train = x[ind_train]
        x_test = x[ind_test]
        x_validate = x[ind_validate]

        y_train = y[ind_train]
        y_test = y[ind_test]
        y_validate = y[ind_validate]

        info_train = info[ind_train]
        info_test = info[ind_test]
        info_validate = info[ind_validate]

        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train.copy(), info_validate, info_test.copy(), columns

