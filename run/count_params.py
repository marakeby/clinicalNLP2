from copy import deepcopy
import pandas as pd
from data.data_access import Data
from model.model_factory import get_model
from model.torch_models.bert_cnn import CNN_Over_BERT
from model.torch_models.bert_linear import Linear_Over_BERT
from model.torch_models.bert_rnn import RNN_Over_BERT
from transformers import TrainingArguments

data_params=  dict(id='progression',
              type= 'updated_label',
              params = {'outcome': 'progression','text': 'NARR+IMPRESS','training_split':9, 'split_path': 'splits_new', 'cloud':False } )

training_args_tuned = TrainingArguments(
    output_dir ='.',
    logging_dir='.',
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=50,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_steps=10)

tiny_bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
mini_bert_model_name= 'google/bert_uncased_L-4_H-256_A-4'
med_bert_model_name= 'google/bert_uncased_L-8_H-512_A-8'
base_bert_model_name= 'google/bert_uncased_L-12_H-768_A-12'
clinical_bert = "emilyalsentzer/Bio_ClinicalBERT"
longformer = 'allenai/longformer-base-4096'

models = [tiny_bert_model_name, mini_bert_model_name, med_bert_model_name, base_bert_model_name, clinical_bert, longformer]
model_ids = ['tiny', 'mini', 'med', 'base', 'clinical', 'longformer']
bert_model_name = tiny_bert_model_name

classifier_params_cnn = dict(nhid=120, output_dim=2, nfilters=120, filter_sizes=[3,5], dropout=0.2)
classifier_params_linear = dict(nhid=120, output_dim=2, dropout=0.2)
classifier_params_rnn = dict(nhid=120,n_layers=1,  output_dim=2, bidirectional=True, dropout=0.2)

bert_template= {
    'type': 'bert',
    'id': 'BERT-tiny',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=CNN_Over_BERT,
                   classifier_params=classifier_params_cnn,
                   training_args=training_args_tuned)
}

all_model_params= {}
for m, id in zip(models,model_ids) :
    bert_model = deepcopy(bert_template)
    bert_model['params']['bert_model_name'] = m
    model = get_model(bert_model)
    trainable_parmas_no, total_parmas_no = model.count_params()
    print(trainable_parmas_no, total_parmas_no)
    all_model_params[id] = [trainable_parmas_no, total_parmas_no]

print(pd.DataFrame(all_model_params))