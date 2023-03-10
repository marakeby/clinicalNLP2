# __author__ = haitham elmarakeby
from file_browser_testing import get_models
from os.path import join, exists
from config_path import TEST_RESULTS_PATH, PLOTS_PATH
from os import listdir
import pandas as pd
import numpy as np

from utils.evaluate import evalualte


# f1 maximization
def expect_f1(y_prob, thres):
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()
    return 2 * tp / (2 * tp + fp + fn)


def optimal_threshold(y_prob):
    y_prob = np.sort(y_prob)[::-1]
    f1s = [expect_f1(y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(f1s)]
    return thres, f1s

def filter_model_dirs(Task, tuned, models, size=None):
    if size is None:
        size = ['base', 'NA']
    else:
        size= size+["NA"]
    print (size)
    if type(tuned) == list:
        tuned.append('NA')
    else:
        tuned = [tuned, 'NA']

    if not type(Task) ==list:
        Task = [Task]
    response_dirs = all_dirs[all_dirs.Task.isin(Task)]
    response_dirs = response_dirs[response_dirs.Model.isin(models)].copy()
    response_dirs = response_dirs[response_dirs.Tuned.isin(tuned)]
    response_dirs = response_dirs[response_dirs.Size.isin(size)]
    print(response_dirs)
    print(response_dirs[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    return response_dirs


def read_predictions(dirs_df):
    model_dict={}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        model = row.Model
        dir_ = join(TEST_RESULTS_PATH, dir_)
        prediction_file = [join(dir_,f) for f in listdir(dir_) if '0_testing.csv' in f][0]
        pred_df = pd.read_csv(prediction_file)

        if max_f1:
            prediction_file_train = [join(dir_, f) for f in listdir(dir_) if '0_traing.csv' in f][0]
            pred_df_train = pd.read_csv(prediction_file_train)
            y_pred_score_train = pred_df_train['pred_score']
            th, f1s = optimal_threshold(y_pred_score_train)
            print('{}  Threshold {} F1 {}'.format(model, th, max(f1s)))
            y_pred = pred_df['pred_score'] >= th
            pred_df['pred'] = y_pred
        model_dict[model] = pred_df
    return model_dict


number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', precision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )
all_models = ['BERT', 'BERT (tuned)', 'clinical BERT', 'CNN', 'tfidf', 'Longformer']
model_mapping = {'BERT (tuned)': 'BERT (tuned)',  'BERT':'BERT', 'clinical BERT': 'clinical BERT', 'tfidf': 'TF-IDF', 'CNN': 'CNN', 'Longformer':'Longformer'}


all_dirs= get_models()


# task = 'progression'
task = 'response'
max_f1 = True

models= ['BERT', 'longformer', 'clinical BERT', 'CNN', 'TF-IDF']
size = ['base', 'mini', 'med', 'tiny', 'NA']
dirs_df = filter_model_dirs(Task=task, tuned=[False, True], models=models, size=size)

for i, row in dirs_df.iterrows():
    if row.Tuned == True:
        row.Model = 'DFCI-ImagingBERT'
dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]+'-'+ dirs_df["Task"]
dirs_df.Model = dirs_df.Model.str.replace('-NA', '')
model_dict = read_predictions(dirs_df)

scores= {}
for i, k in enumerate(model_dict.keys()):
    df = model_dict[k]
    y_test = df['y']
    y_pred_score = df['pred_score']
    y_pred = df['pred']
    # y_pred = y_pred_score>=0.5
    print(k)
    if k in model_mapping.keys():
        model_name = model_mapping[k]
    else:
        model_name = k
    score = evalualte(y_test, y_pred, y_pred_score =y_pred_score)
    scores[model_name] = score

scores_df = pd.DataFrame(scores)
scores_df =scores_df.T
scores_df.columns  = [cols_map[c] for c in scores_df.columns]

scores_df = scores_df.round(2)
save_name = task
if max_f1:
    save_name= save_name + '_max_f1'
scores_df.to_csv(join(PLOTS_PATH, 'table_{}.csv'.format(save_name)))