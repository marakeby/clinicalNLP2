import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

non_files= ['tiny_all_scores.csv', 'mini_all_scores.csv', 'med_all_scores.csv', 'base_all_scores.csv']

frozen_files=['any_cancer_one_split_BERT_cnn_sizes_tiny_frozen_Jan-17_07-36.csv',
'any_cancer_one_split_BERT_cnn_sizes_mini_frozen_Jan-17_07-45.csv',
'any_cancer_one_split_BERT_cnn_sizes_med_frozen_Jan-17_08-12.csv',
'any_cancer_one_split_BERT_cnn_sizes_base_frozen_Jan-17_10-27.csv']

number_patients= [884,592,214,103,68, 35]

dirname = join(dirname, 'gcp_results')
non_dirname = join(dirname, 'nonfrozen')
frozen_dirname = join(dirname, 'frozen')

frozen_dfs=[]
for f in frozen_files:
    df = pd.read_csv(join(frozen_dirname, f))
    frozen_dfs.append(df)

non_dfs=[]
for f in non_files:
    df = pd.read_csv(join(non_dirname, f))
    non_dfs.append(df)

models=['Tiny BERT', 'Mini BERT', 'Med BERT', 'Base BERT' ]

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
        plt.savefig(join(dirname, fname))
        plt.close()
