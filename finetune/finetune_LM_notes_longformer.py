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
longformer_model = 'allenai/longformer-base-4096'

models_dict= {'mini': bert_model_mini, 'med': bert_model_med, 'tiny': bert_model_tiny, 'base':bert_model_base, 'longformer': longformer_model}

train_filename = 'train_text_imaging_only.txt'
valid_filename = 'valid_text_imaging_only.txt'

input_dir = 'gs://profile-notes/geekfest_files/unlabeled_data/' #directory where input data is stored
base_dir = '/home/haithamelmarakeby/pretrained_models_truncated' #used save trained model, checkpoints, and logs

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
        item_tokenized = self.tokenizer.batch_encode_plus([item], max_length= self.max_length, padding='max_length',
                                                  truncation=False, return_special_tokens_mask=True, pad_to_max_length = True)

        trucated_tokens = {key: truncate(val[0], self.max_length - 1) for key, val in item_tokenized.items()}
        ret = {key: torch.tensor(val) for key, val in trucated_tokens.items()}

        return ret

    def __len__(self):
        return len(self.text)

def run(bert_model_name='tiny', dataset = 'valid', max_len=512):
    sub_dir = join(base_dir, bert_model_name)
    model_name = models_dict[bert_model_name]
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name)

    if dataset == 'valid':
        input_filename= valid_filename
        sub_dir = join(sub_dir, 'validation')
    else:
        input_filename= train_filename
        sub_dir = join(sub_dir, 'train')

    output_dir = join(sub_dir, 'output')
    logging_dir = join(sub_dir, 'log')
    saving_dir = join(sub_dir, 'model')

    train_data =  pd.read_csv(join(input_dir,input_filename) )
    train_data = train_data[train_data.text.str.len() > 50]

    # tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_len=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_len=max_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    input_dataset = TorchDataset(train_data['text'].values, tokenizer, max_length=max_len)

    training_args = TrainingArguments(
        output_dir= output_dir,
        logging_dir =logging_dir,  
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size= 4,
        fp16=True,
        save_steps=300_000,
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

    bert_model.save_pretrained(saving_dir)
    print ('saved_file {}'.format(saving_dir))

if __name__=="__main__":
    #use validartion data only 
    run(bert_model_name='longformer', dataset = 'train', max_len=1024)