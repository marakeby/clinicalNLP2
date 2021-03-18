from transformers import AutoModelForMaskedLM,  AutoTokenizer
# model_name= 'google/bert_uncased_L-2_H-128_A-2'
# model_name= 'google/bert_uncased_L-12_H-768_A-12'
# tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_len=512)
# tokenizer.save_pretrained('/home/haithamelmarakeby/pretrained_models/tiny_bert_tuned_notes')
# tokenizer.save_pretrained('/home/haithamelmarakeby/pretrained_models/tiny_bert_tuned_notes_valid')
# tokenizer.save_pretrained('/home/haithamelmarakeby/pretrained_models/base_bert_tuned_notes')

#longformer
model_name= 'allenai/longformer-base-4096'
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_len=1024)
tokenizer.save_pretrained('/home/haithamelmarakeby/pretrained_models/longformer_tuned_notes')