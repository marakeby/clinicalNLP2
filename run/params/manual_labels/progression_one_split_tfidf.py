from model.builders.builders import  build_full_network
import os
filename = os.path.basename(__file__)
vocab_size = 500
data ={'type': 'manual_label' , 'params': {'outcome': 'progression'}}
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
models = [nn,
        {'type': 'svc', 'id': 'Support Vector Machine', 'params': {'kernel': 'linear', 'C': 0.01, 'probability': True}},
        {'type': 'sgd', 'id': 'L2 Logistic Regression', 'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01}},
        {'type': 'random_forest', 'id': 'Random Forest', 'params': {'max_depth': 10, 'n_estimators': 50}},
        {'type': 'adaboost', 'id': 'Adaptive Boosting', 'params': {}},
    ]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True}}
