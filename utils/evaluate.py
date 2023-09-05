from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import sys

def evalualte(y_test, y_pred, y_pred_score =None):
    accuracy = accuracy_score(y_test, y_pred)
    if y_pred_score is None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    matthews_corrcoef = metrics.matthews_corrcoef(y_test, y_pred)
    from sklearn.metrics import average_precision_score
    aupr = average_precision_score(y_test, y_pred_score)
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = precision
    score['auc'] = auc
    score['f1'] = f1
    score['aupr'] = aupr
    score['recall'] = recall
    score['matthews_corrcoef'] = matthews_corrcoef
    return score


import numpy as np
def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = metrics.recall_score(y_true=labels, y_pred=pred)
    precision = metrics.precision_score(y_true=labels, y_pred=pred)
    f1 = metrics.f1_score(y_true=labels, y_pred=pred)    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    filename= sys.argv[1:]
    df = pd.read_csv(filename)
    evalualte(df['truth'], df['pred'], df['score'])