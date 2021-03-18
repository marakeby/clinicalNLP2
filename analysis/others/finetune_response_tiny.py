import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )


original_files = ['response_one_split_BERT_cnn_sizes_tiny_frozen_Jan-27_15-36']

# tuned_files = ['response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-10_13-09']
tuned_files = ['response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46']
# tuned_files = ['response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46']

number_patients= [884,592,214,103,68, 35]

# '/Users/haithamelmarakeby/PycharmProjects/clinicalNLP2/analysis/gcp_results/updated_labels/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-10_13-09'
tuned_dirname = join(dirname, 'gcp_results/updated_labels/tuned_bert_cnn_frozen')
original_direname= join(dirname, 'gcp_results/updated_labels/bert_cnn_arch_size_frozen')

tuned_dfs=[]
for f in tuned_files:
    f = join(tuned_dirname, f)
    df = pd.read_csv(join(f, 'all_scores.csv'))
    tuned_dfs.append(df)

original_dfs=[]
for f in original_files:
    f = join(original_direname, f)
    df = pd.read_csv(join(f, 'all_scores.csv'))
    original_dfs.append(df)

models=['Tiny BERT']

for c in cols_map.keys():
    plt.figure()
    for original_df, tuned_df, model in zip(original_dfs, tuned_dfs, models):
        plt.figure()
        x= number_patients
        y2= tuned_df[c].values
        y1= original_df[c].values
        plt.plot(x,y1, '.-')
        plt.plot(x,y2, '.-')
        
        plt.legend(['original BERT', 'Tuned BERT'])
        plt.xlabel('number of patients')
        plt.ylabel(cols_map[c])
        fname= 'Tuned_{}-{}.png'.format(model,cols_map[c])
        saving_dir = join(dirname, 'plots')
        plt.savefig(join(saving_dir, fname))
        plt.close()
