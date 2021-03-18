import pandas as pd
from os.path import dirname, join
import numpy as np
from transformers import AutoModelForMaskedLM,  AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import torch
torch.manual_seed(0)


bert_model_tiny = 'google/bert_uncased_L-2_H-128_A-2'
bert_model_mini = 'google/bert_uncased_L-4_H-256_A-4'
bert_model_med = 'google/bert_uncased_L-8_H-512_A-8'
bert_model_base = 'google/bert_uncased_L-12_H-768_A-12'

# model_ext = 'base'
model_ext = 'med'
model_name = bert_model_med
# checkpoint_path = '/home/haithamelmarakeby/logs/tiny_bert_tuned_notes_output_valid/checkpoint-40000'
checkpoint_path = '/home/haithamelmarakeby/logs/med_bert_tuned_notes_output/checkpoint-320000'
bert_tiny = AutoModelForMaskedLM.from_pretrained(checkpoint_path)

input_dir = 'gs://profile-notes/geekfest_files/unlabeled_data/'

train_filename = 'train_text_imaging_only.txt'
valid_filename = 'valid_text_imaging_only.txt'


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        item = self.text[idx]
        item_tokenized = self.tokenizer(item, truncation=True, padding=True, return_special_tokens_mask=True)
        ret = {key: torch.tensor(val) for key, val in item_tokenized.items()}
        return ret

    def __len__(self):
        return len(self.text)
    

def run(valid):
    if valid == True:
        filename= join(input_dir,valid_filename)
        output_dir = "/home/haithamelmarakeby/logs/{}_bert_tuned_notes_output_valid".format(model_ext)
        logging_dir = "/home/haithamelmarakeby/logs/{}_bert_tuned_notes_log_valid".format(model_ext),
        saving_dir = '/home/haithamelmarakeby/pretrained_models/{}_bert_tuned_notes_valid'.format(model_ext)
    else:
        filename= join(input_dir,train_filename)
        output_dir = "/home/haithamelmarakeby/logs/{}_bert_tuned_notes_output".format(model_ext)
        logging_dir = "/home/haithamelmarakeby/logs/{}_bert_tuned_notes_log".format(model_ext),
        saving_dir = '/home/haithamelmarakeby/pretrained_models/{}_bert_tuned_notes'.format(model_ext)

    train_data =  pd.read_csv(filename)
    train_data = train_data[train_data.text.str.len() > 50]

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_len=512)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    input_dataset = TorchDataset(train_data['text'].values, tokenizer)

    model = bert_tiny
    training_args = TrainingArguments(
        output_dir= output_dir,
        logging_dir =logging_dir,  
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=40_000,
#         save_total_limit=2,
#         max_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=input_dataset,
    #     prediction_loss_only=True,
    )

    trainer.train()

    
    bert_tiny.save_pretrained(saving_dir)
    print ('saved_file {}'.format(saving_dir))

if __name__=="__main__":
    #use validartion data only 
    run(valid=False)