from copy import copy, deepcopy

from transformers import TrainingArguments, IntervalStrategy, EarlyStoppingCallback
import os
from os.path import join, splitext
from model.torch_models.bert_cnn import CNN_Over_BERT
from model.torch_models.bert_linear import Linear_Over_BERT
from model.torch_models.bert_rnn import RNN_Over_BERT
from utils.evaluate import compute_metrics
from config_path import LOG_PATH


training_splits = [0,1,2,3,4,5, 6, 7, 8, 9]
number_reports = [11182, 9068, 6563, 3887, 2600, 1456, 1072, 750, 495, 172]
number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]


filename = os.path.basename(__file__)
fname = splitext(filename)[0]

d_base = dict(id='progression',
              type= 'updated_label',
              params = {
                        'outcome': 'progression',
                        'text': 'NARR+IMPRESS',
                        'training_split':0, 'cloud':True,
                   'split_path': 'splits_new' 
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
            remove_numbers=True,
            remove_single_chars=True,
            remove_multiple_sapces=True)
           )


training_args = TrainingArguments(
        output_dir=join(LOG_PATH,f'DFCI_BERT/frozen/{fname}'),  # output directory
        num_train_epochs=20,  # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=join(LOG_PATH,f'DFCI_BERT/frozen/{fname}'),  # directory for storing logs
        logging_steps=10,
    
       save_steps= 100,
       evaluation_strategy = IntervalStrategy.STEPS, # "steps"
       eval_steps = 100, # Evaluation and Save happens every 100 steps
       save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
       learning_rate=2e-5,
       metric_for_best_model = 'eval_f1',
       load_best_model_at_end=True,
       # fp16=True
    )



classifier_params_cnn = dict(nhid=120, output_dim=2, nfilters=120, filter_sizes=[3,5], dropout=0.2)
classifier_params_linear = dict(nhid=120, output_dim=2, dropout=0.2)
classifier_params_rnn = dict(nhid=120,n_layers=1,  output_dim=2, bidirectional=True, dropout=0.2)


tiny_bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'
mini_bert_model_name= 'google/bert_uncased_L-4_H-256_A-4'
med_bert_model_name= 'google/bert_uncased_L-8_H-512_A-8'
base_bert_model_name= 'google/bert_uncased_L-12_H-768_A-12'
#pretrained (tuned) DFCI_BERT
tiny_tuned = '/home/jupyter/pretrained_models/bert_tiny_tuned'
base_tuned = '/home/jupyter/pretrained_models/bert_base_tuned'

clinical_bert = "emilyalsentzer/Bio_ClinicalBERT"

bert_model_name = base_tuned

bert_cnn = {
    'type': 'bert',
    'id': 'DFCI_BERT_CNN',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=False,
                   classifier=CNN_Over_BERT,
                   classifier_params=classifier_params_cnn,
                   training_args=training_args,
                 # params for early stopping
                  compute_metrics=compute_metrics,
                  callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
                  )
}

bert_linear = {
    'type': 'bert',
    'id': 'DFCI_BERT_Linear',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=False,
                   classifier=Linear_Over_BERT,
                   classifier_params=classifier_params_linear,
                   training_args=training_args,
                # params for early stopping
                  compute_metrics=compute_metrics,
                  callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
                  )
}
bert_rnn = {
    'type': 'bert',
    'id': 'DFCI_BERT_RNN',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=False,
                   classifier=RNN_Over_BERT,
                   classifier_params=classifier_params_rnn,
                   training_args=training_args,
                 # params for early stopping
                  compute_metrics=compute_metrics,
                  callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
                  
                  )
}


models = [bert_cnn, bert_linear, bert_rnn]


max_length = 512

features = { 'type' : 'bert_tokenizer', 'params': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
feature_selection =[]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'testing'}}
