from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from data.manual_labels.data_reader import ManualLabels
from transformers import  Trainer, TrainingArguments
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
params_files = []

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_dataset(encodings,labels, type='tf'):
    if type=='tf':
        dic= (encodings,labels) #not tested
        return tf.data.Dataset.from_tensor_slices(dic)
    else:
        return TorchDataset(encodings, labels)

bert_mini= 'google/bert_uncased_L-4_H-256_A-4'
bert_tiny= 'google/bert_uncased_L-2_H-128_A-2'

model_name = bert_tiny
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModel.from_pretrained(bert_tiny)
# model = AutoModelForSequenceClassification.from_pretrained(bert_tiny)


data =[{'id':'any_cancer_IMPRESS','type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text': 'IMPRESS'}}]
data = ManualLabels(**data[0]['params'])
x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()


x_train_encoded = tokenizer(list(x_train), max_length=120, truncation=True, padding=True)
x_val_encoded = tokenizer(list(x_validate), max_length=120, truncation=True, padding=True)
x_test_encoded = tokenizer(list(x_test), max_length=120, truncation=True, padding=True)

train_dataset = get_dataset(x_train_encoded,y_train, type='torch')
val_dataset = get_dataset(x_test_encoded,y_validate, type='torch')
test_dataset = get_dataset(x_test_encoded,y_test, type='torch')

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# with training_args.strategy.scope():
model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

for name, param in model.named_parameters():
    print (name)
    if 'classifier' not in name: # classifier layer
        param.requires_grad = False

trainer.train()
test_predictions = trainer.predict(test_dataset)
print (test_predictions.predictions.argmax(axis=1))

print (test_predictions.predictions.shape)