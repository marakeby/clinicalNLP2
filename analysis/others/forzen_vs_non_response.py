import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

non_dirname = join(dirname, 'gcp_results/updated_labels/bert_cnn_arch_size')
frozen_dirname = join(dirname, 'gcp_results/updated_labels/bert_cnn_arch_size_frozen')

frozen_files=['response_one_split_BERT_cnn_sizes_tiny_frozen_Jan-27_15-36',
'response_one_split_BERT_cnn_sizes_mini_frozen_Jan-27_15-45']
# ['response_one_split_BERT_cnn_sizes_base_frozen_Jan-27_18-16',
# 'response_one_split_BERT_cnn_sizes_med_frozen_Jan-27_16-11',
# ]
non_files=['response_one_split_BERT_cnn_sizes_tiny_Jan-27_11-57', 
'response_one_split_BERT_cnn_sizes_mini_Jan-27_12-13']

number_patients= [884,592,214,103,68, 35]


frozen_dfs=[]
for f in frozen_files:
    f = join(frozen_dirname, f)
    df = pd.read_csv(join(f, 'all_scores.csv'))
    frozen_dfs.append(df)

non_dfs=[]
for f in non_files:
    f = join(non_dirname, f)
    df = pd.read_csv(join(f, 'all_scores.csv'))
    non_dfs.append(df)

models=['Tiny BERT', 'Mini BERT' ]

for c in cols_map.keys():
    plt.figure()
    for non_df, frozen_df, model in zip(non_dfs, frozen_dfs, models):
        plt.figure()
        x= number_patients
        y1= frozen_df[c].values
        y2= non_df[c].values
        plt.plot(x,y1, '.-')
        plt.plot(x,y2, '.-')
        
        plt.legend(['Frozen BERT', 'non Frozen BERT'])
        plt.xlabel('number of patients')
        plt.ylabel(cols_map[c])
        fname= '{}-{}.png'.format(model,cols_map[c])
        saving_dir = join(dirname, 'plots')
        plt.savefig(join(saving_dir, fname))
        plt.close()
