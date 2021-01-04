import pandas as pd
import numpy as np

from os.path import join, dirname

dir_path = dirname(dirname(__file__))

input_dir = join(dir_path, 'medical_notes/input/radiology_impressions_derived_data')

filename = join(input_dir, 'manual_label_analysis.csv')
label_analysis = pd.read_csv(filename)

label_analysis.info()

label_analysis = label_analysis.assign(imaging_text=label_analysis.NARR_TXT + ' ' + label_analysis.IMPRESS_TXT)
label_analysis['imaging_text'] = label_analysis.imaging_text.str.replace(r'\r\n', ' ')
label_analysis = label_analysis.drop_duplicates(subset='imaging_text')
label_analysis = label_analysis[label_analysis['imaging_text'].str.contains("it has been imported") == False]
print(label_analysis.info())



# get training/validation/test splits
training_mrns = pd.read_csv(join(input_dir, 'training_mrns.csv'))
validation_mrns = pd.read_csv(join(input_dir, 'validation_mrns.csv'))
test_mrns = pd.read_csv(join(input_dir, 'truetest_mrns.csv'))

training_data = label_analysis[label_analysis['DFCI_MRN'].isin(training_mrns.DFCI_MRN)]
validation_data = label_analysis[label_analysis['DFCI_MRN'].isin(validation_mrns.DFCI_MRN)]
test_data = label_analysis[label_analysis['DFCI_MRN'].isin(test_mrns.DFCI_MRN)]


print(training_data.shape)
print (validation_data.shape)
print (test_data.shape)

number_patients = np.geomspace(10, training_mrns.shape[0], 20)
number_patients = [int(s) for s in number_patients]


samples = training_mrns.copy()

patients = []
for i, n, in enumerate(number_patients[::-1]):
    if i > 0:
        samples = samples[['DFCI_MRN']].sample(n=n, random_state=2008)
    training_data_subsampled = training_data[training_data['DFCI_MRN'].isin(samples.DFCI_MRN)].copy()
    filename = 'training_mrns_{}.csv'.format(i)
    # training_data_subsampled.to_csv(join('processed',filename))
    number_of_selected_reports = training_data_subsampled.shape[0]
    patients.append(training_data_subsampled.DFCI_MRN.unique())
    number_of_selected_patients = len(training_data_subsampled.DFCI_MRN.unique())

    print ('filename {} # patients {} , # of reports {} '.format(filename, number_of_selected_patients,
                                                                number_of_selected_reports))
