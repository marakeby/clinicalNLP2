import pandas as pd
from os.path import join
import numpy as np

from config_path import DATA_PATH

def read_file_gc(filename, fs):
    print(filename)
    with fs.open(filename) as f:
        label_analysis = pd.read_csv(f)
    return label_analysis

processed_dir = join(DATA_PATH, 'updated_labels')

class UpdatedLabels():
    def __init__(self, split_path, outcome = 'any_cancer', text = 'NARR+IMPRESS', training_split=0, cloud=True):

        self.training_split = training_split
        self.training_file= 'training_mrns_{}.csv'.format(self.training_split)
        self.split_path = join(processed_dir, split_path)
        #which dataset to use for training {0: original training, 1, 2, 3..: decreasing sizes of training splits}
        if cloud:
            import gcsfs
            fs = gcsfs.GCSFileSystem(project='vanallen-cnlp')
            
            updated_labels_filename = 'gs://vanallen-cnlp/profile-notes/geekfest_files/geekfest_files_labeled_imaging_reports_manual_dedup_10-8-19.feather'
            # get training and validation patients
            label_analysis = pd.read_feather(updated_labels_filename)
            base_= 'gs://vanallen-cnlp/profile-notes/radiology-impressions-derived-data'
            self.validation_mrns = pd.read_csv(join(base_, 'validation_mrns.csv'))
            self.testing_mrns = pd.read_csv(join(base_, 'truetest_mrns.csv'))
            # self.training_mrns = pd.read_csv(join(base_, 'training_mrns.csv'))
            
            self.training_mrns = pd.read_csv(join(self.split_path, self.training_file))
            
        else:
            input_dir = join(DATA_PATH, 'updated_labels/input')
            filename= join(input_dir, 'geekfest_labeled_imaging_reports_manual_dedup_10-8-19.feather.txt')
            label_analysis = pd.read_feather(filename)

            # get training and validation patients
            self.training_mrns = pd.read_csv(join(self.split_path, self.training_file))
            self.validation_mrns = pd.read_csv(join(input_dir, 'validation_mrns.csv'))
            self.testing_mrns = pd.read_csv(join(input_dir, 'truetest_mrns.csv'))


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

        print ('label_analysis shape {}'.format(label_analysis.shape))
        # drop duplicate reports
        label_analysis = label_analysis.drop_duplicates(subset='imaging_text')
        print ('label_analysis shape adter removing duplication {}'.format(label_analysis.shape))
        # drop outside scans
#         label_analysis = label_analysis.query('imaging_text.str.contains("it has been imported") == False')
        ind = label_analysis.imaging_text.str.contains('it has been imported')
        label_analysis = label_analysis[~ind]
        print ('label_analysis shape adter removing imported {}'.format(label_analysis.shape))
        # logging.info(label_analysis.info())

        x = label_analysis
        x = x.assign(response=np.where(x.redcap_resp_prog == 1, 1, 0))
        x = x.assign(progression=np.where(x.redcap_resp_prog == 4, 1, 0))

        info = x[['DFCI_MRN', 'ehr_scan_date', 'PROC_DESCR']]

        if outcome =='any_cancer':
            y = x['any_cancer'].map(int).copy()
        elif outcome =='response':
            y = x['response'].map(int).copy()
        elif outcome =='progression':
            y = x['progression'].map(int).copy()

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

        ind_train = info['DFCI_MRN'].isin(self.training_mrns.DFCI_MRN)
        ind_validate = info['DFCI_MRN'].isin(self.validation_mrns.DFCI_MRN)
        ind_test = info['DFCI_MRN'].isin(self.testing_mrns.DFCI_MRN)

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

