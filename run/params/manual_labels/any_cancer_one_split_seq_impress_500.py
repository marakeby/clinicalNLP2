from model.builders.builders import build_simple
import os
filename = os.path.basename(__file__)
data ={'type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text':'IMPRESS'}}
pre = {'type' : None}

vocab_size = 2000
# report_max_length = 1000 # number of words per report
report_max_length = 500 # number of words per report
embedding_dims = 150
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

nn_pathway = {
    'type': 'nn',
    'id':'Convolution Neural Network',
    'params':
    {
        'build_fn': build_simple,
        'model_params': {
            'w_reg': 0.001,
            'dropout': 0.3,
            'optimizer': 'Adam',
            'activation': 'relu',
            'vocab_size': vocab_size,
            'input_maxlen': report_max_length,
            'embedding_dims': embedding_dims,
            'kernel_size': 3,
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
                'batch_size':50 ,
                'save_name':filename ,
            },
    },
'selection':  {},

}
# mode = {'count', 'freq', 'tfidf', 'binary'}
features = { 'type' : 'keras_tokenizer', 'parmas': {'mode': 'seq', 'vocab_size': vocab_size, 'pad_length': report_max_length}}
feature_selection =[]
models = [nn_pathway]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True}}
