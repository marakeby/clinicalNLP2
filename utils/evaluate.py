import logging
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import sys

def evalualte(y_test, y_pred, y_pred_score =None):
    accuracy = accuracy_score(y_test, y_pred)
    # score_train = accuracy_score(y_train, y_pred_train)
    if y_pred_score is None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # auc2 = metrics.roc_auc_score(y_test, y_pred_score)
    f1 = metrics.f1_score(y_test, y_pred)
    percision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    logging.info( metrics.classification_report(y_test, y_pred))
    # print '--score: ', accuracy, 'percision: ', percision,'auc1: ', auc, 'f1: ', f1
    from sklearn.metrics import average_precision_score
    aupr = average_precision_score(y_test, y_pred_score)
    logging.info( '--accuracy: {0:.2f} percision: {1:.2f} auc: {2:.2f} f1: {3:.2f} aupr {4:.2f}'.format(accuracy, percision, auc, f1, aupr) )
    print ('--------------------------------------------')
    score = {}
    score['accuracy'] = accuracy
    score['percision'] = percision
    score['auc'] = auc
    score['f1'] = f1
    score['aupr'] = aupr
    score['recall'] = recall
    # logging.info(score)
    # score['aupr'] = aupr
    return score

if __name__ == "__main__":
    filename= sys.argv[1:]
    df = pd.read_csv(filename)
    evalualte(df['truth'], df['pred'], df['score'])