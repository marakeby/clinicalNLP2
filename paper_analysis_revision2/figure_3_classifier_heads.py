import os
import numpy as np
import pandas as pd
from os.path import join, exists, basename
from os import makedirs
from matplotlib import pyplot as plt
from config_path import PLOTS_PATH, TEST_RESULTS_PATH
from paper_analysis_revision2.data_utils import optimal_threshold
from paper_analysis_revision2.file_browser import get_models
from utils.evaluate import evalualte

base_dir = ''
np.random.seed(1234)
# base_dir = join(TEST_RESULTS_PATH, base_dir)
log_dir = join(TEST_RESULTS_PATH, 'classifier/frozen')

all_dirs= get_models()

# def filter_model_dirs(Task, tuned, models, size=None, frozen = None):
#
#
#     if size is None:
#         size = ['base', 'NA']
#     else:
#         size= size+["NA"]
#
#     if frozen is None:
#         frozen = [False, 'NA']
#     else:
#         if not type(frozen) == list:
#             frozen = [frozen]
#         frozen= frozen+['NA']
#
#     print (size)
#     if type(tuned) == list:
#         tuned.append('NA')
#     else:
#         tuned = [tuned, 'NA']
#     response_dirs = all_dirs[all_dirs.Task == Task]
#     response_dirs = response_dirs[response_dirs.Model.isin(models)].copy()
#     response_dirs = response_dirs[response_dirs.Tuned.isin(tuned)]
#     response_dirs = response_dirs[response_dirs.Size.isin(size)]
#     response_dirs = response_dirs[response_dirs.Frozen.isin(frozen)]
#     print(response_dirs)
#     print(response_dirs[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
#     return response_dirs

def get_all_scores(target_dir, max_f1):
    testing_prediction_files = [join(target_dir, f) for f in sorted(os.listdir(target_dir)) if '_testing.csv' in f]
    training_prediction_files = [join(target_dir, f) for f in sorted(listdir(target_dir)) if '_traing.csv' in f]

    all_scores_dict = {}

    for tst_file, train_file in zip(testing_prediction_files, training_prediction_files):
        # print (basename(tst_file), basename(train_file))
        f_name = tst_file.replace('_testing.csv', '')
        f_name_tr = train_file.replace('_traing.csv', '')
        assert (f_name == f_name_tr)
        print(basename(f_name))

        pred_df = pd.read_csv(tst_file)

        if max_f1:
            pred_df_train = pd.read_csv(train_file)
            y_pred_score_train = pred_df_train['pred_score']
            th, f1s = optimal_threshold(y_pred_score_train)
            y_pred = pred_df['pred_score'] >= th
            pred_df['pred'] = y_pred

        all_scores = evalualte(pred_df['y'], pred_df['pred'], y_pred_score=pred_df['pred_score'])
        all_scores_dict[basename(f_name)] = all_scores
    df = pd.DataFrame(all_scores_dict).T
    return df

def plot_classifier(task, frozen):

    models = ['BERT', 'CNN', 'TF-IDF']
    title = task.capitalize()
    # frozen = [True, False]
    dirs_df = filter_model_dirs(Task=task, tuned=True, models=models, frozen=frozen)
    # dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    dirs_df.Model = dirs_df["Model"].astype(str)
    dirs_df.Model = dirs_df.Model.str.replace('-NA','')

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            row.Model = 'DFCI-ImagingBERT'
        if row.Frozen == True:
            row.Model = row.Model + ' (Frozen)'

    # if frozen:
    #     dash ='.-'
    #     log_dir = join(TEST_RESULTS_PATH, 'classifier/frozen')
    #     if task == 'progression_tuned':
    #         # log_dir_ = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Nov-15_15-36')
    #         # log_dir_ = join(log_dir, 'progression_DFCI_BERT_sizes_base_frozen_May-25_09-59')
    #         log_dir_ = join(log_dir, 'progression_DFCI_BERT_sizes_base_frozen_May-30_06-38')
    #         title= 'Progression (Frozen)'
    #     elif task == 'response_tuned':
    #         # log_dir_ = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Nov-16_02-49')
    #         # log_dir_ = join(log_dir, 'response_DFCI_BERT_sizes_base_frozen_May-25_13-39')
    #         log_dir_ = join(log_dir, 'response_DFCI_BERT_sizes_base_frozen_May-30_03-34')
    #         title = 'Response (Frozen)'
    #     else:
    #         raise
    # else:
    #     dash = '.--'
    #     log_dir = join(TEST_RESULTS_PATH, 'classifier/unfrozen')
    #     if task == 'progression_tuned':
    #         # log_dir_ = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Nov-15_15-36')
    #         # log_dir_ = join(log_dir, 'progression_DFCI_BERT_sizes_base_unfrozen_May-25_16-42')
    #         # log_dir_ = join(log_dir, 'progression_DFCI_BERT_sizes_base_unfrozen_linear_try1_May-28_02-35')
    #         log_dir_ = join(log_dir, 'progression_DFCI_BERT_sizes_base_unfrozen_linear_try1_May-29_17-50')
    #         title = 'Progression'
    #     elif task == 'response_tuned':
    #         # log_dir_ = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Nov-16_02-49')
    #         # log_dir_ = join(log_dir, 'response_DFCI_BERT_sizes_base_unfrozen_May-25_22-22')
    #         # log_dir_ = join(log_dir, 'response_DFCI_BERT_sizes_base_unfrozen_linear_try1_May-27_21-41')
    #         log_dir_ = join(log_dir, 'response_DFCI_BERT_sizes_base_unfrozen_linear_try1_May-29_09-58')
    #         title = 'Response'
    #     else:
    #         raise
    # frozen_str = 'frozen' if frozen else 'unfrozen'
    # saving_dir = join(PLOTS_PATH, 'figure3_classifers_{}_{}'.format(task, frozen_str))
    frozen_str= 'frozen' if frozen else 'unfrozen'
    # tuned_str= 'DFCI_BERT' if tuned else 'raw_BERT'
    saving_dir = join(PLOTS_PATH, 'figure3_DFCI_BERT_classifers2/{}/{}'.format(task, frozen_str))

    if not exists(saving_dir):
        makedirs(saving_dir)

    if not exists(saving_dir):
        os.makedirs(saving_dir)
    dirs_df.to_csv(join(saving_dir, 'models.csv'))
    df = get_all_scores(dirs_df, max_f1=max_f1, )


    # filename = join(log_dir_,'all_scores.csv' )
    # df = pd.read_csv(filename, index_col=0)
    # print ( df.head())
    cnn_ind = df.index.str.contains('CNN')
    rnn_ind = df.index.str.contains('RNN')
    linear_ind = df.index.str.contains('Linear')

    cnn_df = df[cnn_ind].copy()
    rnn_df = df[rnn_ind].copy()
    linear_df = df[linear_ind].copy()

    dfs = [cnn_df,rnn_df, linear_df ]
    concat_df = pd.concat(dfs)
    concat_df.to_csv(join(saving_dir, 'models.csv'))
    legend=[ 'CNN', 'RNN', 'Linear' ]
    if frozen:
        legend = [l+' (Frozen)' for l in legend]

    cols_map=dict(accuracy='Accuracy', precision='Precision', auc='AUROC', f1='F1',aupr='AUPRC', recall= 'Recall' )

    # number_patients= [884,592,214,103,68, 35]
    number_patients = [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]


    for c in cols_map.keys():
        plt.figure()
        ax = plt.subplot(111)
        for df in dfs:
            x= number_patients
            y= df[c].values
            print (c)
            ax.plot(x,y, dash)
        plt.legend(legend, loc='lower right')
        plt.xlabel('Number of patients')
        plt.ylabel(cols_map[c])
        plt.ylim((0.4,1))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fname= '{}.png'.format(c)
        print(fname)
        plt.title(title)
        plt.savefig(join(saving_dir, fname))


if __name__ == '__main__':
    plot_classifier(task = 'progression_tuned', frozen=True)
    plot_classifier(task = 'progression_tuned', frozen=False)

    plot_classifier(task = 'response_tuned', frozen=True)
    plot_classifier(task = 'response_tuned', frozen=False)

    # plot_classifier(task = 'response_tuned')






