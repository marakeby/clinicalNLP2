import os
from os.path import join, basename
import pandas as pd
from config_path import TEST_RESULTS_PATH
from os import listdir
import numpy as np

from utils.evaluate import evalualte


def read_predictions(dirs_df,max_f1,  _class= 0):
    model_dict={}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        model = row.Model
        dir_ = join(TEST_RESULTS_PATH, dir_)
        prediction_file = [join(dir_,f) for f in listdir(dir_) if '0_testing.csv' in f][0]
        pred_df = pd.read_csv(prediction_file)

        if _class ==1:
            pred_df['pred_score'] = 1 - pred_df['pred_score']
            pred_df['pred']  = 1- pred_df['pred']
            pred_df['y'] = 1- pred_df['y']

        if max_f1 and not 'zeroshot' in model:
            prediction_file_train = [join(dir_, f) for f in listdir(dir_) if '0_traing.csv' in f][0]
            pred_df_train = pd.read_csv(prediction_file_train)

            if _class == 1:
                pred_df_train['pred_score'] = 1 - pred_df_train['pred_score']
                pred_df_train['pred'] = 1 - pred_df_train['pred']
                pred_df_train['y'] = 1 - pred_df_train['y']

            y_pred_score_train = pred_df_train['pred_score']
            th, f1s = optimal_threshold(y_pred_score_train)
            y_pred = pred_df['pred_score'] >= th
            pred_df['pred'] = y_pred

        model_dict[model] = pred_df
    return model_dict

def optimal_threshold(y_prob):
    y_prob = np.sort(y_prob)[::-1]
    f1s = [expect_f1(y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(f1s)]
    return thres, f1s

# f1 maximization
def expect_f1(y_prob, thres):
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()
    return 2 * tp / (2 * tp + fp + fn)


def filter_model_dirs(all_dirs, Task, tuned, models, size=None, frozen = None):


    if size is None:
        size = ['base', 'NA',  'xxl']
    else:
        size= size+["NA"]

    if frozen is None:
        frozen = [False, 'NA']
    else:
        if not type(frozen) == list:
            frozen = [frozen]
        frozen= frozen+['NA']

    print (size)
    if type(tuned) == list:
        tuned.append('NA')
    else:
        tuned = [tuned, 'NA']
    response_dirs = all_dirs[all_dirs.Task == Task]
    response_dirs = response_dirs[response_dirs.Model.isin(models)].copy()
    response_dirs = response_dirs[response_dirs.Tuned.isin(tuned)]
    response_dirs = response_dirs[response_dirs.Size.isin(size)]
    response_dirs = response_dirs[response_dirs.Frozen.isin(frozen)]
    print(response_dirs)
    print(response_dirs[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    return response_dirs


def get_all_scores(target_dir, max_f1):
    testing_prediction_files = [join(target_dir, f) for f in sorted(os.listdir(target_dir)) if '_testing.csv' in f]
    training_prediction_files = [join(target_dir, f) for f in sorted(os.listdir(target_dir)) if '_traing.csv' in f]

    all_scores_dict = {}
    if not max_f1:
        for tst_file in testing_prediction_files:
            f_name = tst_file.replace('_testing.csv', '')
            pred_df = pd.read_csv(tst_file)
            all_scores = evalualte(pred_df['y'], pred_df['pred'], y_pred_score=pred_df['pred_score'])
            all_scores_dict[basename(f_name)] = all_scores
    else:
        for tst_file, train_file in zip(testing_prediction_files, training_prediction_files):
            # print (basename(tst_file), basename(train_file))
            f_name = tst_file.replace('_testing.csv', '')
            f_name_tr = train_file.replace('_traing.csv', '')
            assert (f_name == f_name_tr)
            print(basename(f_name))

            pred_df = pd.read_csv(tst_file)
            pred_df_train = pd.read_csv(train_file)

            y_pred_score_train = pred_df_train['pred_score']
            th, f1s = optimal_threshold(y_pred_score_train)
            y_pred = pred_df['pred_score'] >= th
            pred_df['pred'] = y_pred

            all_scores = evalualte(pred_df['y'], pred_df['pred'], y_pred_score=pred_df['pred_score'])
            all_scores_dict[basename(f_name)] = all_scores
    df = pd.DataFrame(all_scores_dict).T
    return df