from model.builders.builders import build_simple
import os

from model.torch_models.cnn import CNN
from model.torch_models.cnn1d import CNN1D

filename = os.path.basename(__file__)
data ={'id':'any_cancer', 'type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text':'NARR+IMPRESS'}}
data=[data]
# pre = {'type' : None}
pre = {'type' : 'clean_text'}

# vocab_size = 2000
vocab_size = 2000
# report_max_length = 1000 # number of words per report
# report_max_length = 500 # number of words per report
report_max_length = 500 # number of words per report
embedding_dims = 150
# filters = 250
# kernel_size = 3
# hidden_dims = 250
epochs = 5
batch_size=50
dropout =0.3

nn_torch = {
    'type': 'nn_torch',
    'id':'Convolution Neural Network',
    'params': dict(model=CNN,
                   model_params=dict(
                       output_dim=2,
                       vocab_size=vocab_size+4,
                       embedding_dim=embedding_dims,
                       dropout=dropout,
                       filter_sizes=[3],
                       n_filters = 250,
                       hidden_dims=100
                       ),
                   training_parmas = dict(batch_size=batch_size, epochs=epochs)),
    'selection':  {},
}
# mode = {'count', 'freq', 'tfidf', 'binary'}
features = { 'type' : 'torch_tokenizer', 'parmas': {'mode': 'seq', 'vocab_size': vocab_size, 'pad_length': report_max_length}}
feature_selection =[]
models = [nn_torch]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True}}
