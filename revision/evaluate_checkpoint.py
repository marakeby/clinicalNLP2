import torch
torch.manual_seed(0)
import os
import pandas as pd
from os.path import dirname, join
import numpy as np
import time
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import AutoModelForMaskedLM,  AutoTokenizer


# model_name = '/home/haithamelmarakeby/pretrained_models_truncated/base/train/output/checkpoint-40000'
# bert_model = AutoModelForMaskedLM.from_pretrained(model_name)


train_filename = 'train_text_imaging_only.txt'
valid_filename = 'valid_text_imaging_only.txt'
input_dir = 'gs://profile-notes/geekfest_files/unlabeled_data/'

print ('loading data')
train_data =  pd.read_csv(join(input_dir,train_filename) )
valid_data =  pd.read_csv(join(input_dir,valid_filename) )

train_data = train_data[train_data.text.str.len() > 50]
valid_data = valid_data[valid_data.text.str.len() > 50]

print(train_data.shape, valid_data.shape)

max_len = 512
bert_model_base = 'google/bert_uncased_L-12_H-768_A-12'
tokenizer = AutoTokenizer.from_pretrained(bert_model_base, truncation=False, padding=True, max_len=max_len)

def truncate(x, n):
    ret = [x[0]] + x[-n:]
    return ret

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, idx):
        item = self.text[idx]
        # item_tokenized = self.tokenizer(item, truncation=True, padding=True, return_special_tokens_mask=True)
        item_tokenized = self.tokenizer.batch_encode_plus([item], max_length= self.max_length, padding='max_length',
                                                  truncation=False, return_special_tokens_mask=True, pad_to_max_length = True)

        trucated_tokens = {key: truncate(val[0], self.max_length - 1) for key, val in item_tokenized.items()}
        ret = {key: torch.tensor(val) for key, val in trucated_tokens.items()}
        
#         item_tokenized = self.tokenizer(item, truncation=True, padding='max_length', return_special_tokens_mask=True)
#         ret = {key: torch.tensor(val) for key, val in item_tokenized.items()}
        
#         print (idx, len(ret['input_ids']),len(ret['token_type_ids']), len(ret['attention_mask']), len(ret['special_tokens_mask']))
        return ret

    def __len__(self):
        return len(self.text)
    

input_dataset = TorchDataset(train_data['text'].values, tokenizer, max_length=max_len)
valid_dataset = TorchDataset(valid_data['text'].values, tokenizer, max_length=max_len)
small_dataset = TorchDataset(valid_data['text'].values[0:1000], tokenizer, max_length=max_len)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

def get_trainer(model_name):
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    output_dir = '.'
    logging_dir = '.'
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    epoch_steps = round(train_data.shape[0]/per_device_train_batch_size) #roughly 1 epoch

    training_args = TrainingArguments(
        output_dir= output_dir,
        logging_dir =logging_dir,  
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size= per_device_train_batch_size,
        per_device_eval_batch_size =per_device_eval_batch_size, 
        save_steps = epoch_steps , 
    #         save_steps=40_000,

        do_eval = False,
        do_train = False,
        evaluation_strategy='steps',
        eval_steps = epoch_steps
    #         save_strategy = 'epoch'
    #         save_total_limit=2,
    #         max_steps=100
    )

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=input_dataset,
        eval_dataset = valid_dataset
    #     prediction_loss_only=True,
    )
    return trainer


rootdir = '/home/haithamelmarakeby/pretrained_models_truncated/base/train/output/output'

print ('collecting models')
def get_mode_names():
    model_names = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)
            model_names.append(d)
    
    return sorted(model_names)

t0 = time.time()
model_names= get_mode_names()
train_metrics = []
for model_path in model_names:
    model_name = os.path.basename(model_path)
    print(model_name)
    print('building trainer')
    trainer = get_trainer(model_path)
    print('evaluation')
    train_metric = trainer.evaluate(valid_dataset)
    del trainer
    train_metric['model'] = model_name
    print(model_name, train_metric)
    train_metrics.append(train_metric)


t1 = time.time()

total = t1-t0
print('time spent', total)
df = pd.DataFrame(train_metrics)
print(df)
df.to_csv('base_scores_validation.csv')