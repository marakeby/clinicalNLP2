import logging
import numpy as np
# import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import softmax
from transformers import BertModel
from transformers import Trainer
import torch
from model.bert_model_utils import TorchDataset
from model.torch_models.bert_classifier import BertSequenceClassificationModel


def get_dataset(encodings,labels, type='tf'):
    if type=='tf':
        dic= (encodings,labels) #not tested
        return tf.data.Dataset.from_tensor_slices(dic)
    else:
        return TorchDataset(encodings, labels)

def get_trainer(model, training_args, train_dataset, val_dataset=None, callbacks=None, compute_metrics = None):

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        
        compute_metrics = compute_metrics,
        callbacks = callbacks
    )
    return trainer

class Bert_Torch_Model(BaseEstimator):
    def __init__(self, bert_model_name, freez_bert, classifier, classifier_params, training_args, compute_metrics=None, callbacks=None):
        self.bert_model_name  = bert_model_name
        self.classifier  = classifier
        self.freez_bert  = freez_bert
        self.classifier_params  = classifier_params
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.training_args = training_args
        self.callbacks= callbacks
        self.compute_metrics = compute_metrics
        self.loss ='sigmoid'


    def set_params(self, sk_params):
        self.params = sk_params
        pass

    def get_params(self, deep=False):
        return self.params

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('self.model_name ', self.bert_model_name )
        bert= BertModel.from_pretrained(self.bert_model_name)
        self.model = BertSequenceClassificationModel(bert, self.freez_bert, self.classifier, **self.classifier_params )
        self.model.to(device)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_parmas_no = count_parameters(self.model.bert )
        logging.info('Trainable params no {}'.format(trainable_parmas_no))
        train_dataset = get_dataset(X_train, y_train, type='torch')
        if X_val is not None:
            val_dataset = get_dataset(X_val, y_val, type='torch')
        else:
            val_dataset = train_dataset

        self.trainer = get_trainer(self.model,self.training_args, train_dataset, val_dataset, callbacks=self.callbacks, compute_metrics=self.compute_metrics)
        self.trainer.train()
        return self

    def predict(self, X_test):
        prediction_scores = self.predict_proba(X_test)
        prediction=prediction_scores.argmax(axis=1)
        return prediction

    def predict_proba(self, X_test):
        n_samples = len(X_test['input_ids'])
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


