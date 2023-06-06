import os
import numpy as np
from os.path import join, exists
from sklearn import metrics
from config_path import  PLOTS_PATH
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from paper_analysis_revision2.plot_utils import plot_auc_all, plot_auc_bootstrap, plot_prc_all, plot_confusion_matrix
from paper_analysis_revision2.data_utils import filter_model_dirs, read_predictions
from paper_analysis_revision2.file_browser import get_models


np.random.seed(1234)

def plot_figure1_compare_model_sizes(task='response',tuned=False, frozen = False):

    title = task.capitalize()
    models = ['longformer', 'BERT']
    size = ['base', 'mini', 'med', 'tiny']
    model_mapping= { 'tfidf': 'TF-IDF', 'longformer':'Longformer'}
    dirs_df = filter_model_dirs(all_dirs, Task= task, tuned=tuned, models=models, size=size, frozen= frozen)
    dirs_df.Model.replace(model_mapping, inplace=True)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-'+ dirs_df["Size"]
    dirs_df.Model.replace({'Longformer-base': 'Longformer'}, inplace=True)
    print(dirs_df.Model)

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
              row.Model = 'DFCI-ImagingBERT'
        if row.Frozen:
            row.Model = row.Model+' (Frozen)'
        print(row.Model)

    frozen_str= 'frozen' if frozen else 'unfrozen'
    tuned_str= 'DFCI_BERT' if tuned else 'raw_BERT'
    filename = 'figure1/{}/{}/{}'.format(task, tuned_str, frozen_str)

    saving_dir = join(PLOTS_PATH, filename)
    if not exists(saving_dir):
        os.makedirs(saving_dir)

    filename= join(saving_dir,'models.csv')
    dirs_df.to_csv(filename)

    model_dict = read_predictions(dirs_df, max_f1=max_f1, _class=_class)

    keys= ['Longformer', 'BERT-base', 'BERT-med', 'BERT-mini', 'BERT-tiny']
    if frozen:
        keys = [k+' (Frozen)' for k in keys]

    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict,  ax, keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    filename= join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)
    plt.close()

    #AUC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict,  ax, keys, sort_auc=False, introduce_line_on='(')
    ax.set_ylabel('AUROC')
    filename = join(saving_dir, '_auc_bootsrtap')
    plt.title(title , fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

    ## PRC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_prc_all(model_dict, ax, keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    filename = join(saving_dir, '_prc')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metrics.average_precision_score, introduce_line_on='(')
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap')
    plt.savefig(filename, dpi=400)
    plt.close()

    #
    metrics_dict = dict(F1=metrics.f1_score, Accuracy=metrics.accuracy_score, Precision=metrics.precision_score,Recall= metrics.recall_score, MCC= metrics.matthews_corrcoef)
    for k, metric_func in metrics_dict.items():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
        plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metric_func, pred_col='pred', introduce_line_on='(')
        ax.set_ylabel(k)
        filename = join(saving_dir, '_{}_bootsrtap'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()


    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

    for i, k in enumerate(model_dict.keys()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2), dpi=400)
        df = model_dict[k]
        y_test = df['y']
        y_pred_score = df['pred']
        cnf_matrix = confusion_matrix(y_test, y_pred_score)
        cm = np.array(cnf_matrix)
        classes = ['Negative', 'Positive']
        labels = np.array([['TN', 'FP'], ['FN ', 'TP']])

        plot_confusion_matrix(ax, cm, classes,
                              labels=labels,
                              fontproperties=fontproperties,
                              normalize=True,
                              cmap=plt.cm.Reds)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        plt.subplots_adjust(left=0.15)
        filename = join(saving_dir, '_{}_confusion_matrix'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()


if __name__ == '__main__':
    #Figure 1
    all_dirs = get_models()

    max_f1 = True
    _class = 0  ## _class=1 flips the prediction scores and labels (default = 0)

    plot_figure1_compare_model_sizes(task='response',tuned=False, frozen=True)
    plot_figure1_compare_model_sizes(task='progression',tuned=False, frozen =True)