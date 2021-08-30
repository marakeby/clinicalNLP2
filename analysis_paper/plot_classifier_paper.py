# from setup import saving_dir
import pandas as pd
import os
from os.path import join, exists
from os import makedirs
from matplotlib import pyplot as plt
from config_path import PLOTS_PATH, GCP_RESULTS_PATH

# saving_dir = join(saving_dir, 'classifier')
task = 'progression_tuned'
# task = 'response_tuned'
# base_dir  = os.path.dirname(__file__)
base_dir ='updated_labels'
base_dir = join(GCP_RESULTS_PATH,base_dir )
log_dir = join(base_dir, 'bert_classifier')

log_dir = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Jun-02_18-04')
# log_dir = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Jun-03_05-33')

saving_dir = join(PLOTS_PATH, 'classifers_{}'.format(task))

# saving_dir = PLOTS_PATH
if not exists(saving_dir):
    makedirs(saving_dir)

filename = join(log_dir,'all_scores.csv' )

df = pd.read_csv(filename, index_col=0)
print ( df.head())
cnn_ind = df.index.str.contains('CNN')
rnn_ind = df.index.str.contains('RNN')
linear_ind = df.index.str.contains('Linear')

cnn_df = df[cnn_ind].copy()
rnn_df = df[rnn_ind].copy()
linear_df = df[linear_ind].copy()

print(linear_df)
print(cnn_df)
print(rnn_df)



print (linear_ind)
print (cnn_ind)
print (rnn_df)

dfs = [linear_df, cnn_df, rnn_df]

cols= ['accuracy',	'precision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', precision='Precision', auc='Area under the ROC curve (AUC)', f1='F1',aupr='Area under the precision-recall curve (AUPRC)', recall= 'Recall' )
# files= ['response_tiny.csv', 'response_mini.csv']


number_patients= [884,592,214,103,68, 35]
# # legend=['Tiny BERT', 'Mini BERT', 'Med BERT', 'Base BERT' , 'Longformer']
legend=['Linear', 'CNN', 'RNN' ]


for c in cols_map.keys():
    plt.figure()
    for df in dfs:
        x= number_patients
        y= df[c].values
        print (c)
        plt.plot(x,y, '.-')
    plt.legend(legend)
    plt.xlabel('number of patients')
    plt.ylabel(cols_map[c])
    fname= '{}.png'.format(c)
    print(fname)
    plt.savefig(join(saving_dir, fname))





