from transformers import TrainingArguments
import os

from model.torch_models.bert_rnn import RNN_Over_BERT

filename = os.path.basename(__file__)
data =[{'id':'any_cancer_IMPRESS','type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text': 'NARR+IMPRESS','training_split':0}}]
pre = {'type' : None}
from os.path import join



training_args = TrainingArguments(
        output_dir=join('./results',filename),  # output directory
        num_train_epochs=2,  # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=join('./logs',filename),  # directory for storing logs
        logging_steps=10,
    )

classifier_params = dict(nhid=120,n_layers=1,  output_dim=2, bidirectional=True, dropout=0.2)

bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
bert= {
    'type': 'bert',
    'id': 'BERT',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=RNN_Over_BERT,
                   classifier_params=classifier_params,
                   training_args=training_args)
}

max_length = 512

features = { 'type' : 'bert_tokenizer', 'parmas': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
models = [bert]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'validation'}}
