from transformers import TrainingArguments
import os
filename = os.path.basename(__file__)
data =[{'id':'any_cancer_IMPRESS','type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text': 'NARR+IMPRESS','training_split':10}}]
pre = {'type' : None}


bert_model_name= 'google/bert_uncased_L-2_H-128_A-2'

training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

bert= {
    'type': 'bert',
    'id': 'BERT',
    'params': dict(model_name=bert_model_name, training_args=training_args)
}

max_length = 512

features = { 'type' : 'bert_tokenizer', 'parmas': dict(model_name= bert_model_name, truncation= True, padding=True, max_length= max_length)}
# features = { 'type' : 'bert_tokenizer', 'parmas': {'model_name': bert_model_name, 'truncation': True, 'padding': True}}
feature_selection =[]
models = [bert]
pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset':'validation'}}
