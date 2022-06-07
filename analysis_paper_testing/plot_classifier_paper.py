# from setup import saving_dir
import pandas as pd
import os
from os.path import join, exists
from os import makedirs
from matplotlib import pyplot as plt
from config_path import PLOTS_PATH, TEST_RESULTS_PATH

# saving_dir = join(saving_dir, 'classifier')
# task = 'progression_tuned'
task = 'response_tuned'


base_dir = ''
base_dir = join(TEST_RESULTS_PATH, base_dir)
log_dir = join(base_dir, 'updated_labels')
log_dir = join(log_dir, 'bert_classifier')

if task == 'progression_tuned':
    # log_dir = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Sep-21_11-08')
    log_dir = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Nov-15_15-36')
    title= 'Progression'
elif task == 'response_tuned':
    # log_dir = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Sep-21_23-01')
    log_dir = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Nov-16_02-49')
    title = 'Response'

saving_dir = join(PLOTS_PATH, 'figure3_classifers_{}'.format(task))

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

dfs = [cnn_df,rnn_df, linear_df ]
legend=[ 'CNN', 'RNN', 'Linear' ]

cols= ['accuracy',	'precision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', precision='Precision', auc='AUROC', f1='F1',aupr='AUPRC', recall= 'Recall' )
# files= ['response_tiny.csv', 'response_mini.csv']


number_patients= [884,592,214,103,68, 35]
# # legend=['Tiny BERT', 'Mini BERT', 'Med BERT', 'Base BERT' , 'Longformer']


for c in cols_map.keys():
    plt.figure()
    ax = plt.subplot(111)
    for df in dfs:
        x= number_patients
        y= df[c].values
        print (c)
        ax.plot(x,y, '.-')
    plt.legend(legend, loc='lower right')
    plt.xlabel('Number of patients')
    plt.ylabel(cols_map[c])
    plt.ylim((0,1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fname= '{}.png'.format(c)
    print(fname)
    plt.title(title)
    plt.savefig(join(saving_dir, fname))






