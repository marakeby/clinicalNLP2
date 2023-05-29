from copy import copy, deepcopy

from transformers import TrainingArguments,  EarlyStoppingCallback, IntervalStrategy
import os
from os.path import join, splitext
from model.torch_models.bert_cnn import CNN_Over_BERT
from model.torch_models.bert_linear import Linear_Over_BERT
from model.torch_models.bert_rnn import RNN_Over_BERT
from config_path import LOG_PATH
from utils.evaluate import compute_metrics

# training_splits = [0,1,2,3,4,5, 6, 7, 8, 9]
# number_reports = [11182, 9068, 6563, 3887, 2600, 1456, 1072, 750, 495, 172]
# number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]

training_splits = [0]
number_reports = [11182]
number_patients= [884]

filename = os.path.basename(__file__)
fname = splitext(filename)[0]

#------------meta_data---------------------------
meta_data= dict(
     Task='response', 
     Model='BERT', 
     Size='base', 
     Tuned= False, 
     Frozen=False,
     classifier = 'Linear', 
     file=filename
    )

#------------data_params---------------------------
d_base = dict(id=meta_data['Task'],
              type= 'updated_label',
              params = {
                        'outcome': meta_data['Task'],
                        'text': 'NARR+IMPRESS',
                        'training_split':0,
                      'split_path': 'splits_new' 
              }
              )
data=[]
for i in training_splits:
    d = deepcopy(d_base)
    d['id'] = '{}_{}'.format(meta_data['Task'], i)
    d['params']['training_split'] = i
    print (d)
    data.append(d)
print(data)

# ---------preprocessing------------------------------

pre = dict(type= 'clean_text',
            parmas= dict(remove_tag=True,
            remove_special_chars=True,
            remove_numbers=False,
            remove_single_chars=True,
            remove_multiple_sapces=True)
           )


# -----------------TrainingArguments----------------------

training_args = TrainingArguments(
        output_dir=join(LOG_PATH,f'bert/unfrozen/{fname}'),  # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation

        logging_dir=join(LOG_PATH,f'bert/unfrozen/{fname}'),  # directory for storing logs
        logging_steps=300,
       save_steps= 1000000,
       evaluation_strategy = IntervalStrategy.STEPS, # "steps"
       eval_steps = 300, # Evaluation and Save happens every 100 steps
       load_best_model_at_end=False,
        warmup_ratio = 0.3,
    learning_rate=  1e-5
    )

# -------------------BERT models --------------------

classifier_params_cnn = dict(nhid=120, output_dim=2, nfilters=120, filter_sizes=[3,5], dropout=0.2)
classifier_params_linear = dict(nhid=120, output_dim=2, dropout=0.2)
classifier_params_rnn = dict(nhid=120,n_layers=1,  output_dim=2, bidirectional=True, dropout=0.2)


classifier_ = dict(Linear = Linear_Over_BERT,
     CNN= CNN_Over_BERT,
     RNN = RNN_Over_BERT)

classifier_params = dict(Linear = classifier_params_linear,
     CNN= classifier_params_cnn,
     RNN = classifier_params_rnn)


tiny_bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
mini_bert_model_name= 'google/bert_uncased_L-4_H-256_A-4'
med_bert_model_name= 'google/bert_uncased_L-8_H-512_A-8'
base_bert_model_name= 'google/bert_uncased_L-12_H-768_A-12'

#pretrained (tuned) DFCI_BERT
tiny_tuned = '/home/jupyter/pretrained_models/bert_tiny_tuned'
base_tuned = '/home/jupyter/pretrained_models/bert_base_tuned'

bert_size= dict(base = base_bert_model_name, med= med_bert_model_name, mini= mini_bert_model_name, tiny=tiny_bert_model_name)
bert_size_tuned = dict(tiny = tiny_tuned, base = base_tuned)

# ---------------------------------------

bert = {
    'type': 'bert',
    'id': meta_data['Model'],
    'params': dict(bert_model_name = bert_size[meta_data['Size']],
                   freez_bert = meta_data['Frozen'],
                   classifier = classifier_[meta_data['classifier']],
                   classifier_params= classifier_params[meta_data['classifier']],
                   training_args=training_args,
                  
                  # params for early stopping
                  compute_metrics=compute_metrics,
                  # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
                  )
}


# ---------------------------------------

max_length = 512

features = { 'type' : 'bert_tokenizer', 'params': dict(model_name= bert_size[meta_data['Size']], truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
models = [bert ]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'testing'}}
# pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'validation'}}
