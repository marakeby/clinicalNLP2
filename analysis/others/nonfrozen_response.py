import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

dirname = join(dirname, 'gcp_results/updated_labels/bert_cnn_arch_size')

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )
# files= ['tiny_all_scores.csv', 'mini_all_scores.csv', 'med_all_scores.csv', 'base_all_scores.csv', 'long_all_scores.csv']
# files=['any_cancer_one_split_BERT_cnn_sizes_tiny_frozen_Jan-17_07-36.csv',
# 'any_cancer_one_split_BERT_cnn_sizes_mini_frozen_Jan-17_07-45.csv',
# 'any_cancer_one_split_BERT_cnn_sizes_med_frozen_Jan-17_08-12.csv',
# 'any_cancer_one_split_BERT_cnn_sizes_base_frozen_Jan-17_10-27.csv']

files=['response_one_split_BERT_cnn_sizes_mini_Jan-27_12-13',
'response_one_split_BERT_cnn_sizes_tiny_Jan-27_11-57']

number_patients= [884,592,214,103,68, 35]
# legend=['Tiny BERT', 'Mini BERT', 'Med BERT', 'Base BERT' , 'Longformer']
legend=['Tiny BERT', 'Mini BERT', 'Med. BERT', 'Base BERT' ]
legend=legend[::-1]

dfs=[]
for f in files:
    f = join(dirname, f) 
    df = pd.read_csv(join(f, 'all_scores.csv'))
    dfs.append(df)

for c in cols_map.keys():
    plt.figure()
    for df in dfs:
        x= number_patients
        y= df[c].values
        plt.plot(x,y, '.-')
    plt.legend(legend)
    plt.xlabel('number of patients')
    plt.ylabel(cols_map[c])
    fname= '{}.png'.format(c)
    plt.savefig(join(dirname, fname))





