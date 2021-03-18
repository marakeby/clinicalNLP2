import pandas as pd
import os
from os.path import join
from matplotlib import pyplot as plt
dirname  = os.path.dirname(__file__)

dirname = join(dirname, 'gcp_results')
dirname = join(dirname, 'response')

cols= ['accuracy',	'percision',	'auc',	'f1',	'aupr',	'recall']
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )
files= ['response_tiny.csv', 'response_mini.csv']


number_patients= [884,592,214,103,68, 35]
# legend=['Tiny BERT', 'Mini BERT', 'Med BERT', 'Base BERT' , 'Longformer']
legend=['Tiny BERT', 'Mini BERT' ]

dfs=[]
for f in files:
    df = pd.read_csv(join(dirname, f))
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
    print(fname)
    plt.savefig(join(dirname, fname))





