from model.builders.builders import  build_full_network
import os
filename = os.path.basename(__file__)
from copy import copy, deepcopy

vocab_size = 20000
data_base ={'id':'response','type': 'updated_label' , 'params': {'outcome': 'response', 'text': 'NARR+IMPRESS', 'training_split':0}}

training_splits = [0,1,4,6,7,9]
number_reports = [11182,7382,2897,1214, 865, 453 ]
number_patients= [884,592,214,103,68, 35 ]

data=[]

for i in training_splits:
    d = deepcopy(data_base)
    d['id'] = 'any_cancer_{}'.format(i)
    d['params']['training_split'] = i
    print (d)
    data.append(d)

pre = {'type' : None}

# mode = {'count', 'freq', 'tfidf', 'binary'}
features = { 'type' : 'keras_tokenizer', 'parmas': {'mode': 'tfidf', 'vocab_size': vocab_size, 'pad_length': None}}
nn = {
    'type': 'nn',
    'id':'Neural Network',
    'params':
    {
        'build_fn': build_full_network,
        'model_params': {
            'w_reg': [0.001]*2,
            'dropouts': [0.7, 0.5],
            'optimizer': 'Adam',
            'activation': 'relu',
            'n_features': vocab_size,
            'n_hids': [200, 100],
        },
            'fitting_params': {
                'select_best_model': False,
                'monitor': 'val_acc',
                'verbose': 2,
                'epoch': 100,
                'shuffle' : True,
                'batch_size':50 ,
                'save_name':filename,
            },

    }
}

feature_selection =[]
# models = [nn,
#         {'type': 'svc', 'id': 'Support Vector Machine', 'params': {'kernel': 'linear', 'C': 0.01, 'probability': True}},
#         {'type': 'sgd', 'id': 'L2 Logistic Regression', 'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01}},
#         {'type': 'random_forest', 'id': 'Random Forest', 'params': {'max_depth': 10, 'n_estimators': 50}},
#         {'type': 'adaboost', 'id': 'Adaptive Boosting', 'params': {}},
#     ]
models=[nn]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True}}
