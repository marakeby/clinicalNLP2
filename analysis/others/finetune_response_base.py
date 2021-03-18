import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )


original_files = ['response_one_split_BERT_cnn_sizes_base_frozen_Jan-27_18-16']

tuned_files = ['response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29']
# tuned_files = ['response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46']

number_patients= [884,592,214,103,68, 35]

# '/Users/haithamelmarakeby/PycharmProjects/clinicalNLP2/analysis/gcp_results/updated_labels/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-10_13-09'
tuned_dirname = join(dirname, 'gcp_results/updated_labels/tuned_bert_cnn_frozen')
original_direname= join(dirname, 'gcp_results/updated_labels/bert_cnn_arch_size_frozen')

JAMA_file= join(dirname, 'gcp_results/updated_labels/JAMA/response_one_split_sizes_JAMA_Feb-24_11-21')
TFIDF_file = join(dirname, 'gcp_results/updated_labels/tfidf/response_one_split_tfidf_Feb-24_12-31')

TFIDF_data = pd.read_csv(join(TFIDF_file,'all_scores.csv' ))
JAMA_data = pd.read_csv(join(JAMA_file,'all_scores.csv' ))

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

models=['Base BERT']
include_JAMA =True
include_TFIDF= False
for c in cols_map.keys():
    plt.figure()
    for original_df, tuned_df, model in zip(original_dfs, tuned_dfs, models):
        plt.figure()
        x= number_patients
        y2= tuned_df[c].values
        y1= original_df[c].values
        plt.plot(x,y1, '.-')
        plt.plot(x,y2, '.-')
        legend = ['Original BERT', 'Tuned BERT']
        filename =model
        if include_JAMA:
            y3= JAMA_data[c].values
            plt.plot(x, y3, '-.')
            legend.append('JAMA')
            filename=filename+'_JAMA'

        if include_TFIDF:
            y4= TFIDF_data[c].values
            plt.plot(x, y4, '--')
            legend.append('TF-IDF')
            filename = filename + '_TFIDF'

        plt.legend(legend)
        plt.xlabel('number of patients')
        plt.ylabel(cols_map[c])
        fname= 'Tuned_{}-{}.png'.format(filename,cols_map[c])
        saving_dir = join(dirname, 'plots')
        plt.savefig(join(saving_dir, fname))
        plt.close()
