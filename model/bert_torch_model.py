import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification
import tensorflow as tf
from sklearn.utils.extmath import softmax
from model.bert_model_utils import TorchDataset, BertForLongSequenceClassification, CNNOverBertSequenceClassification
from transformers import TFTrainer, TFTrainingArguments, Trainer, TrainingArguments
import logging

def get_dataset(encodings,labels, type='tf'):
    if type=='tf':
        dic= (encodings,labels) #not tested
        return tf.data.Dataset.from_tensor_slices(dic)
    else:
        return TorchDataset(encodings, labels)

def get_trainner(model, training_args, train_dataset, val_dataset=None):

    # training_args = TrainingArguments(
    #     output_dir='./results',  # output directory
    #     num_train_epochs=1,  # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,  # batch size for evaluation
    #     warmup_steps=500,  # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     logging_dir='./logs',  # directory for storing logs
    #     logging_steps=10,
    # )


    # with training_args.strategy.scope():
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #     model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )
    return trainer

class Bert_Torch_Model(BaseEstimator):
    def __init__(self, model_name, training_args):
        self.model_name  = model_name
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.training_args = training_args
        self.loss ='sigmoid'


    def set_params(self, sk_params):
        self.params = sk_params
        pass

    def get_params(self, deep=False):
        return self.params

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # train_dataset = get_dataset(X_train, list(y_train), type='torch')
        print('self.model_name ', self.model_name )
        print (type(X_train))

        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name )
        # self.model = BertForLongSequenceClassification.from_pretrained(self.model_name )
        self.model = CNNOverBertSequenceClassification.from_pretrained(self.model_name )

        # print(self.model)
        for name, param in self.model.named_parameters():
            print (name)
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_parmas_no = count_parameters(self.model )
        logging.info('trainable_parmas_no {}'.format(trainable_parmas_no))
        train_dataset = get_dataset(X_train, y_train, type='torch')
        self.trainer = get_trainner(self.model,self.training_args, train_dataset, train_dataset)
        self.trainer.train()

        return self

    def predict(self, X_test):
        prediction_scores = self.predict_proba(X_test)
        prediction=prediction_scores.argmax(axis=1)
        return prediction

    def predict_proba(self, X_test):
        # test_labels = None
        n_samples = len(X_test.input_ids)
        print('n_samples', n_samples)
        test_labels = np.ones((n_samples,), np.long)
        test_dataset = get_dataset(X_test, test_labels, type='torch')
        prediction_outputs = self.trainer.predict(test_dataset)
        prediction_scores = prediction_outputs.predictions
        prediction_scores = softmax(prediction_scores)

        return prediction_scores

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return accuracy_score(y_test, y_pred)


    def save_model(self, filename):
        self.model.save_weights(filename)


