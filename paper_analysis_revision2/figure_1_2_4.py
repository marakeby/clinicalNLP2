from os.path import join, exists
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix, precision_recall_curve
from paper_analysis.plot_utils import plot_confusion_matrix
from file_browser import get_models
import pandas as pd
from matplotlib import pyplot as plt
import os

from paper_analysis_revision2.data_utils import filter_model_dirs, read_predictions
from paper_analysis_revision2.plot_utils import model_colors
from utils.stat_utils import score_ci
from config_path import  PLOTS_PATH
import numpy as np
from sklearn import metrics
import collections




all_models = ['BERT', 'BERT (tuned)', 'clinical BERT', 'CNN', 'tfidf', 'Longformer']
model_mapping = {'BERT (tuned)': 'DFCI-ImagingBERT',  'BERT':'BERT', 'clinical BERT': 'clinical BERT', 'tfidf': 'TF-IDF', 'CNN': 'CNN', 'Longformer':'Longformer'}


def plot_roc(ax, y_test, y_pred_score, color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    symbol = '-'
    if  'TF-IDF' in label:
        symbol = '--'
    elif 'CNN' in label:
        symbol = '-.'
    ax.plot(fpr, tpr, symbol, label=label + ' (area=%0.2f)' % roc_auc, linewidth=1, color=color)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)

def sort_dict(all_models_dict):
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        sorted_dict[k] = average_auc
        print('model {} , auc= {}'.format(k, average_auc))

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    return sorted_dict


def plot_jitter(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    inds = np.arange(1, len(sums) + 1)
    for i, s in zip(inds, sums.index):
        ind = data[group_col] == s
        n = sum(ind)
        x = data.loc[ind, val_col]
        y = np.array([i - 0.3] * n)
        noise = np.random.normal(0, 0.02, n)
        y = y + noise
        ax.plot(x, y, '.', markersize=5)







def plot_auc_bootstrap_boxplot(all_models_dict, ax):
    all_scores=[]
    names=[]
    xs=[]
    avg_scores=[]
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metrics.roc_auc_score,
                                                     n_bootstraps=1000, seed=123)
        all_scores.append(scores)
        names.append(k)
        xs.append(np.random.normal(i + 1, 0.04, len(scores)))
        avg_scores.append(score)

    sorted_list = [(x, n) for _,x,n in sorted(zip(avg_scores, all_scores, names))]
    all_scores  = [x[0] for x in sorted_list]
    names  = [x[1] for x in sorted_list]
    ax.boxplot(all_scores, labels= names)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    for i, (x, val, clevel, k) in enumerate(zip(xs, all_scores, clevels, names)):
        model_name = model_mapping[k]
        plt.scatter(x, val,marker='.',s=10, color=model_colors[model_name], alpha=0.05)
    plt.ylim(0.8,1.0)
    plt.locator_params(axis="y", nbins=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def plot_auc_all(all_models_dict,ax,  sorting_keys=None, sort_auc=False):
    if sorting_keys is None:
        keys = all_models_dict.keys()
    else:
        keys = sorting_keys

    print ('keys', keys)
    if sort_auc:
        keys = sort_dict(all_models_dict)

    for i, k in enumerate(keys):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        print(k)
        if k in model_mapping.keys():
            model_name= model_mapping[k]
        else:
            model_name= k
        plot_roc(ax, y_test, y_pred_score, color=model_colors[model_name], label=model_name)


def plot_prc_all(all_models_dict,ax,  sorting_keys=None, sort_auc=False):
    if sorting_keys is None:
        keys = all_models_dict.keys()
    else:
        keys = sorting_keys

    if sort_auc:
        keys = sort_dict(all_models_dict)

    for i, k in enumerate(keys):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        if k in model_mapping.keys():
            model_name= model_mapping[k]
        else:
            model_name= k
        from plot_utils import plot_prc
        plot_prc(ax, y_test, y_pred_score, color=model_colors[model_name], fontproperties= fontproperties, label=model_name)


fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
# number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

all_dirs= get_models()

#AUC
def plot_figure4(task='response'):
    models = ['BERT', 'CNN', 'TF-IDF']
    title = task.capitalize()
    frozen = [True, False]
    dirs_df = filter_model_dirs(all_dirs, Task=task, tuned=True, models=models, frozen=frozen)
    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    dirs_df.Model = dirs_df.Model.str.replace('-NA','')

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            row.Model = 'DFCI-ImagingBERT'
        if row.Frozen == True:
            row.Model = row.Model + ' (Frozen)'
    filename = 'figure4/{}/combined'.format(task)
    saving_dir = join(PLOTS_PATH, filename)

    if not exists(saving_dir):
        os.makedirs(saving_dir)
    dirs_df.to_csv(join(saving_dir, 'models.csv'))
    model_dict = read_predictions(dirs_df, max_f1=max_f1, _class=_class)

    print ('model_dict', model_dict.keys())
    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    filename= join(saving_dir, '_auc_tuned')
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)

    ## AUC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, introduce_line_on='(')
    filename = join(saving_dir, '_auc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)
    plt.close()

    ## PRC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_prc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    # saving_dir = join(PLOTS_PATH, 'figure4_model_arch_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc_tuned')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, metric=metrics.average_precision_score, introduce_line_on='(')
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

def save_plots_fig2(model_dict, keys, saving_dir, title):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)

    plot_auc_all(model_dict, ax, sorting_keys=keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    filename = join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, sorting_keys=keys, sort_auc=False, introduce_line_on='(')
    filename = join(saving_dir, '_auc_bootsrtap')
    ax.set_ylabel('AUROC')
    plt.title(title, fontsize=10)
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
    plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metrics.average_precision_score,
                       introduce_line_on='(')
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap')
    # plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

    metrics_dict = dict(F1=metrics.f1_score, Accuracy=metrics.accuracy_score, Precision=metrics.precision_score,
                        Recall=metrics.recall_score)
    for k, metric_func in metrics_dict.items():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
        plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metric_func, pred_col='pred')
        ax.set_ylabel(k)
        filename = join(saving_dir, '_{}_bootsrtap'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()

    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

    for i, k in enumerate(model_dict.keys()):
        if k.endswith('_train'):
            continue
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2), dpi=400)
        df = model_dict[k]
        y_test = df['y']
        y_pred = df['pred']

        cnf_matrix = confusion_matrix(y_test, y_pred)
        cm = np.array(cnf_matrix)
        classes = ['Negative', 'Positive']
        labels = np.array([['TN', 'FP'], ['FN ', 'TP']])

        plot_confusion_matrix(ax, cm, classes,
                              labels=labels,
                              fontproperties=fontproperties,
                              normalize=True,
                              cmap=plt.cm.Reds)

        ax.tick_params(axis=u'both', which=u'both', length=0)
        filename = join(saving_dir, '_{}_confusion_matrix{}'.format(k, 'max_f1'))
        plt.subplots_adjust(left=0.15)
        plt.savefig(filename, dpi=400)
        plt.close()


def plot_figure2_compare_domain_adaptation_combined(task = 'response'):
    title = task.capitalize()
    models = ['BERT', 'clinical BERT']
    dirs_df = filter_model_dirs(Task=task, tuned=[True, False], models=models, frozen=[True, False])

    print (dirs_df)
    print (dirs_df.shape)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    # dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"] + '-' + dirs_df['Frozen'].astype(str)

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
                row.Model = 'DFCI-ImagingBERT'
        if row.Frozen:
            row.Model = row.Model+' (Frozen)'
        print(row.Model)

    filename = 'figure2_domain_adaptation_/{}/compare'.format(task)
    saving_dir = join(PLOTS_PATH,filename )
    if not exists(saving_dir):
        os.makedirs(saving_dir)

    dirs_df.to_csv(join(saving_dir, 'models.csv'))

    model_dict = read_predictions(dirs_df, max_f1=max_f1,  _class=_class)
    keys = ['BERT-base (Frozen)', 'clinical BERT-base (Frozen)', 'DFCI-ImagingBERT', 'DFCI-ImagingBERT (Frozen)']

    save_plots_fig2(model_dict, keys, saving_dir, title)


def plot_figure2_compare_domain_adaptation(task = 'response', frozen=False):
    title = task.capitalize()
    models = ['BERT', 'clinical BERT']
    dirs_df = filter_model_dirs(Task=task, tuned=[True, False], models=models, frozen=frozen)

    print (dirs_df)
    print (dirs_df.shape)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    # dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"] + '-' + dirs_df['Frozen'].astype(str)

    # sorting_keys
    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            # row.Model = row.Model + ' (tuned on DFCI)'
                row.Model = 'DFCI-ImagingBERT'

        if row.Frozen:
            row.Model = row.Model+' (Frozen)'
        print(row.Model)

    frozen_str = 'frozen' if frozen else 'unfrozen'
    filename = 'figure2_domain_adaptation_/{}/{}'.format(task, frozen_str)
    saving_dir = join(PLOTS_PATH,filename )
    if not exists(saving_dir):
        os.makedirs(saving_dir)

    dirs_df.to_csv(join(saving_dir, 'models.csv'))

    model_dict = read_predictions(dirs_df, max_f1=max_f1, _class=_class)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)

    keys = ['BERT-base', 'clinical BERT-base', 'DFCI-ImagingBERT']

    if frozen:
        keys= [k+' (Frozen)' for k in keys]

    save_plots_fig2(model_dict, keys, saving_dir, title)



def plot_compare_seq_length(task='response', tuned=False):

    title = 'Compare Sequnce length Arch.'
    if tuned:
        title =  title  + '(Tuned BERT)'
    else:
        title = title + '(Original BERT)'

    if task =='response':
        title = 'Response: '+title
    elif task == 'progression':
        title = 'Progression: ' + title

    models = ['BERT', 'clinical BERT', 'longformer']
    model_mapping= { 'tfidf': 'TF-IDF', 'longformer':'Longformer'}
    dirs_df = filter_model_dirs(Task=task, tuned=tuned, models=models)
    dirs_df.Model.replace(model_mapping, inplace=True)
    print(dirs_df.Model)
    model_dict = read_predictions(dirs_df, _class=_class)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(PLOTS_PATH, 'seq_length_{}_{}'.format(task, tuned))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename= join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax)
    filename = join(saving_dir, '_auc_bootsrtap')
    plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()


def get_title(task='response',tuned=False):
    if task=='response':
        task ='Response'
    elif task =='progression':
        task ='Progression'

    if tuned:
        tuned='(tuned BERT)'
    else:
        tuned = '(Original BERT)'
    title = '{} {}'.format(task, tuned)
    return title

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
            row.Model = row.Model+'(Frozen)'
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
        keys = [k+'(Frozen)' for k in keys]
    # keys= [ 'BERT-base', 'BERT-med', 'BERT-mini', 'BERT-tiny']
    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict,  ax, keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    # filename = 'figure1_{}_tuned_{}_frozen_{}'.format(task, tuned, frozen)



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
    # plt.title(title, fontsize=10)
    # saving_dir = join(PLOTS_PATH, 'figure1_model_size_{}_{}'.format(task, tuned))
    # if not exists(saving_dir):
    #     os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metrics.average_precision_score)
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap')
    # plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

    #
    metrics_dict = dict(F1=metrics.f1_score, Accuracy=metrics.accuracy_score, Precision=metrics.precision_score,Recall= metrics.recall_score, MCC= metrics.matthews_corrcoef)
    for k, metric_func in metrics_dict.items():
        # print (k , metric_func)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=400)
        plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metric_func, pred_col='pred')
        ax.set_ylabel(k)
        filename = join(saving_dir, '_{}_bootsrtap'.format(k))
        # plt.title(title, fontsize=10)
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
        # plt.title(title, fontsize=10)
        plt.subplots_adjust(left=0.15)
        filename = join(saving_dir, '_{}_confusion_matrix'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()

if __name__ == '__main__':
    #Figure 1
    max_f1 = True
    _class =  0 ## _class=1 flips the prediction scores and labels (default = 0)
    # plot_figure1_compare_model_sizes(task='response',tuned=False, frozen=False)
    # plot_figure1_compare_model_sizes(task='progression',tuned=False, frozen =False)
    # # # #
    plot_figure1_compare_model_sizes(task='response',tuned=False, frozen=True)
    plot_figure1_compare_model_sizes(task='progression',tuned=False, frozen =True)

    #Figure 2
    # plot_figure2_compare_domain_adaptation_combined(task='response')
    # plot_figure2_compare_domain_adaptation_combined(task='progression')
    #
    # plot_figure2_compare_domain_adaptation('progression', frozen=True)
    # plot_figure2_compare_domain_adaptation('response', frozen=True)
    # plot_figure2_compare_domain_adaptation('response', frozen=False)
    # plot_figure2_compare_domain_adaptation('progression', frozen=False)
    #
    # #Figure 4
    plot_figure4('response')
    plot_figure4('progression')
    # #
    # plot_figure4('response', frozen=True)
    # plot_figure4('progression', frozen = True)

'''
Model architecture
	BERT, CNN, TF-IDF
Domain adaptation
	BERT_tuned,  clinical BERT, CNN, TF-IDF
Sequence Length 
	BERT,  Longformer , clinical BERT, CNN, TF-IDF
	BERT_tuned,  Longformer _tuned, clinical BERT, CNN, TF-IDF
Model size
	base, tiny, med, mini
	base, tiny, med, mini (tuned)
Training sizes
'''


