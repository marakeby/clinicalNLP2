from copy import copy, deepcopy

from transformers import TrainingArguments, EarlyStoppingCallback, IntervalStrategy
import os
from os.path import join, splitext
from model.torch_models.bert_cnn import CNN_Over_BERT
from model.torch_models.bert_linear import Linear_Over_BERT
from model.torch_models.bert_rnn import RNN_Over_BERT
from config_path import LOG_PATH
from utils.evaluate import compute_metrics


training_splits = [0,1,2,3,4,5, 6, 7, 8, 9]
number_reports = [11182, 9068, 6563, 3887, 2600, 1456, 1072, 750, 495, 172]
number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]

training_splits=training_splits[9:]
number_reports=number_reports[9:]
number_patients=number_patients[9:]

# training_splits = [0]
# number_reports = [11182]
# number_patients= [884]


filename = os.path.basename(__file__)
d_base = dict(id='response',
              type= 'updated_label',
              params = {
                        'outcome': 'response',
                        'text': 'NARR+IMPRESS',
                        'training_split':0,
                  'split_path': 'splits_new' 
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

fname = splitext(filename)[0]



training_args_tuned = TrainingArguments(
        output_dir=join(LOG_PATH,f'frozen/{fname}'),  # output directory
        num_train_epochs=10,  # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=join(LOG_PATH,f'frozen/{fname}'),  # directory for storing logs
        
        # do_train=True,
        # do_eval=True,

        overwrite_output_dir=True,
        # save_steps= 100000000,
        # save_total_limit= 1,
    #params for early stop
    #    gradient_accumulation_steps=1,
    #    logging_steps=10,
    #    evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    #    eval_steps = 20, # Evaluation and Save happens every 50 steps
    #    save_steps=20,
    #    save_total_limit = 5, # Only last 10 models are saved. Older ones are deleted.
    #    learning_rate=2e-5,
    #    push_to_hub=False,
    #    metric_for_best_model = 'eval_f1',
    # greater_is_better =True,
    #    load_best_model_at_end=True
    
#         fp16=True
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

bert_model_name = base_tuned

bert_tuned= {
    'type': 'bert',
    'id': 'BERT_Tuned',
    'params': dict(bert_model_name=bert_model_name,
                   freez_bert=False,
                   classifier=Linear_Over_BERT,
                   classifier_params=classifier_params_linear,
                   training_args=training_args_tuned,
                  
                   # params for early stopping
                  compute_metrics=compute_metrics,
                  # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
                  callbacks = None

                  )
}



max_length = 512

features = { 'type' : 'bert_tokenizer', 'params': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
models = [bert_tuned ]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'testing'}}
