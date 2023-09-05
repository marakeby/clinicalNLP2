import torch
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import softmax
from torch import optim, nn
from torch.utils.data import DataLoader
import time
import logging
# from barbar import Bar



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __getitem__(self, idx):
        item={}
        item['text'] = torch.tensor(self.texts[idx])
        item['label'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TorchModel(BaseEstimator):
    def __init__(self, model, model_params, training_parmas):

        self.model_params=model_params
        assert 'batch_size' in training_parmas, logging.error('missing batch_size params')
        assert 'epochs' in training_parmas, logging.error('missing epochs params')

        self.batch_size = training_parmas['batch_size']
        self.epochs = training_parmas['epochs']

        self.training_params= training_parmas
        self.model = model(**model_params)
        logging.info(self.model)
        logging.info('Trainable params {}'.format(count_parameters(self.model)))


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        model = self.model.to(device)
        criterion = criterion.to(device)

        train_iterator = DataLoader(TextDataset(X_train, y_train), batch_size=self.batch_size, shuffle=False)
        valid_iterator=None
        optimizer = optim.Adam(model.parameters())
        self.train_model(self.model, self.epochs, train_iterator, valid_iterator, optimizer, criterion)
        return self

    def train_model(self, model, n_epochs, train_iterator, valid_iterator, optimizer, criterion):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            logging.info(' Train acc {}, Train loss {}'.format( train_acc, train_loss ))
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')


    def predict(self, X_test):
        prediction_scores = self.predict_proba(X_test)
        prediction = prediction_scores.argmax(axis=1)
        return prediction

    def predict_proba(self, X_test):
        self.model.eval()
        print(type(X_test))
        n_samples = X_test.shape[0]
        print('n_samples', n_samples)
        test_labels = np.ones((n_samples,), np.long)

        test_iter = DataLoader(TextDataset(X_test, test_labels), batch_size=self.batch_size, shuffle=False)
        prediction_outputs=[]
        with torch.no_grad():
            for batch in test_iter:
                predictions = self.model(batch['text']).squeeze(1)
                prediction_outputs.append(predictions)

        prediction_scores = torch.cat(prediction_outputs).numpy()
        prediction_scores = softmax(prediction_scores)

        return prediction_scores

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename):
        self.model.save_weights(filename)



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for i, batch in enumerate(Bar(iterator)):
        optimizer.zero_grad()
        # print (type(batch['text']))
        # print('...')
        predictions = model(batch['text'])
        loss = criterion(predictions, batch['label'])
        acc = categorical_accuracy(predictions, batch['label'])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])