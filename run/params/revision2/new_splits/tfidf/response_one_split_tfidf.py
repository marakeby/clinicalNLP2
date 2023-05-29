# from model.builders.builders import  build_full_network
import os
filename = os.path.basename(__file__)
from copy import copy, deepcopy

vocab_size = 20000
data_base ={'id':'response','type': 'updated_label' , 'params': {'outcome': 'response', 'text': 'NARR+IMPRESS', 'training_split':0, 'split_path': 'splits_new'}}

training_splits = [0,1,2,3,4,5, 6, 7, 8, 9]
number_reports = [11182, 9068, 6563, 3887, 2600, 1456, 1072, 750, 495, 172]
number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]

data=[]

for i in training_splits:
    d = deepcopy(data_base)
    d['id'] = 'response_{}'.format(i)
    d['params']['training_split'] = i
    print (d)
    data.append(d)

pre = {'type' : None}
features = { 'type' : 'sk_tokenizer', 'params': {}}
feature_selection =[]
models=[{'type': 'sgd', 'id': 'L2 Logistic Regression', 'params': {'loss': 'log', 'penalty': 'elasticnet'}}]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset': 'testing'}}
