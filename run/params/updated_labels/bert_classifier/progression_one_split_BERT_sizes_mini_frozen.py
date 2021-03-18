from copy import copy, deepcopy

from transformers import TrainingArguments
import os
from os.path import join
from model.torch_models.bert_cnn import CNN_Over_BERT
from model.torch_models.bert_linear import Linear_Over_BERT
from model.torch_models.bert_rnn import RNN_Over_BERT

training_splits = [0,1,4,6,7,9]
number_reports = [11182,7382,2897,1214, 865, 453 ]
number_patients= [884,592,214,103,68, 35 ]


filename = os.path.basename(__file__)
d_base = dict(id='progression',
              type= 'updated_label',
              params = {
                        'outcome': 'progression',
                        'text': 'NARR+IMPRESS',
                        'training_split':0, 'cloud':True
              }
              )
data=[]
for i in training_splits:
    d = deepcopy(d_base)
    d['id'] = 'progression_{}'.format(i)
    d['params']['training_split'] = i
    print (d)
    data.append(d)
print(data)

#preprocessing
pre = dict(type= 'clean_text',
            parmas= dict(remove_tag=True,
            remove_special_chars=True,
            remove_numbers=False,
            remove_single_chars=True,
            remove_multiple_sapces=True)
           )


training_args = TrainingArguments(
        output_dir=join('/home/haithamelmarakeby/results',filename),  # output directory
        num_train_epochs=10,  # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=join('/home/haithamelmarakeby/logs',filename),  # directory for storing logs
        logging_steps=10,
    )



# bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
bert_model_name= 'google/bert_uncased_L-4_H-256_A-4'
# bert_model_name= 'google/bert_uncased_L-8_H-512_A-8'
# bert_model_name= 'google/bert_uncased_L-12_H-768_A-12'

classifier_params_cnn = dict(nhid=120, output_dim=2, nfilters=120, filter_sizes=[3,5], dropout=0.2)
classifier_params_linear = dict(nhid=120, output_dim=2, dropout=0.2)
classifier_params_rnn = dict(nhid=120,n_layers=1,  output_dim=2, bidirectional=True, dropout=0.2)


bert_cnn = {
    'type': 'bert',
    'id': 'BERT_CNN',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=CNN_Over_BERT,
                   classifier_params=classifier_params_cnn,
                   training_args=training_args)
}

bert_linear = {
    'type': 'bert',
    'id': 'BERT_Linear',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=Linear_Over_BERT,
                   classifier_params=classifier_params_linear,
                   training_args=training_args)
}
bert_rnn = {
    'type': 'bert',
    'id': 'BERT_RNN',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=RNN_Over_BERT,
                   classifier_params=classifier_params_rnn,
                   training_args=training_args)
}


models = [bert_cnn, bert_linear, bert_rnn]


max_length = 512

features = { 'type' : 'bert_tokenizer', 'parmas': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'validation'}}
