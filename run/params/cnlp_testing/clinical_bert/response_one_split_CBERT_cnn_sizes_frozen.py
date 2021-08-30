from copy import copy, deepcopy

from transformers import TrainingArguments
import os
from os.path import join
from model.torch_models.bert_cnn import CNN_Over_BERT

training_splits = [0,1,4,6,7,9]
number_reports = [11182,7382,2897,1214, 865, 453 ]
number_patients= [884,592,214,103,68, 35 ]


filename = os.path.basename(__file__)
d_base = dict(id='response',
              type= 'updated_label',
              params = {
                        'outcome': 'response',
                        'text': 'NARR+IMPRESS',
                        'training_split':0
              }
              )
data=[]
for i in training_splits:
    d = deepcopy(d_base)
    d['id'] = 'response_{}'.format(i)
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
        output_dir=join('/home/haithamelmarakeby/testing/results',filename),  # output directory
        num_train_epochs=20,  # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=join('/home/haithamelmarakeby/testing/logs',filename),  # directory for storing logs
        logging_steps=10,
    )


classifier_params = dict(nhid=120, output_dim=2, nfilters=120, filter_sizes=[3,5], dropout=0.2)

# bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
# bert_model_name = '/home/haithamelmarakeby/pretrained_models/tiny_bert_tuned_notes_valid'
# bert_model_name = '/home/haithamelmarakeby/pretrained_models/tiny_bert_tuned_notes'
bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"

bert= {
    'type': 'bert',
    'id': 'BERT',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=True,
                   classifier=CNN_Over_BERT,
                   classifier_params=classifier_params,
                   training_args=training_args)
}

max_length = 512

features = { 'type' : 'bert_tokenizer', 'params': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
models = [bert]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'testing'}}
