from model.builders.JAMA_builders import any_cancer_JAMA
import os
from copy import copy, deepcopy
filename = os.path.basename(__file__)
data_base ={'id':'progression','type': 'updated_label' , 'params': {'outcome': 'progression', 'text': 'NARR+IMPRESS', 'training_split':0}}

# training_splits = [0,1,4,6,7,9]
# number_reports = [11182,7382,2897,1214, 865, 453 ]
# number_patients= [884,592,214,103,68, 35 ]

training_splits = [0,1,2,3,4,5, 6, 7, 8, 9]
number_reports = [11182, 9068, 6563, 3887, 2600, 1456, 1072, 750, 495, 172]
number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]

data=[]

for i in training_splits:
    d = deepcopy(data_base)
    d['id'] = 'progression_{}'.format(i)
    d['params']['training_split'] = i
    print (d)
    data.append(d)

pre = {'type' : None}

vocab_size = 20000
report_max_length = 1000 # number of words per report
embedding_dims = 300
filters = 300
kernel_size = 10
hidden_dims = 250
epochs = 2
batch_size = 32
dropout=0.2
activation = 'relu'
optimizer = 'Adam'


cnn = {
    'type': 'nn',
    'id':'CNN',
    'params':
    {
        'build_fn': any_cancer_JAMA,
        'model_params': {
#             'w_reg': 0.001,
            'dropout': dropout,
            'optimizer': optimizer,
            'activation': activation,
            'vocab_size': vocab_size,
            'input_maxlen': report_max_length,
            'embedding_dims': embedding_dims,
            'kernel_size': kernel_size,
            'hidden_dims': hidden_dims,
            'filters': filters,
            'strides': 1,
        },

            'fitting_params': {
                'select_best_model': False,
                'monitor': 'val_acc',
                'verbose': 2,
                 'epoch': epochs,
                'shuffle' : True,
                'batch_size':batch_size ,
                'save_name':filename ,
            },
    },
'selection':  {},

}
# mode = {'count', 'freq', 'tfidf', 'binary'}
features = { 'type' : 'keras_tokenizer', 'params': {'mode': 'seq', 'vocab_size': vocab_size, 'pad_length': report_max_length}}
feature_selection =[]
models = [cnn]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset': 'testing'}}
