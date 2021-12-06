from copy import deepcopy
from os.path import join, exists

from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix
from analysis_paper_testing.plot_utils import plot_confusion_matrix
from file_browser_testing import get_models
import pandas as pd
from matplotlib import pyplot as plt
import os
from os import listdir

from analysis.stat_utils import score_ci

# base_dir = os.path.dirname(__file__)
from config_path import TEST_RESULTS_PATH, PLOTS_PATH

import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score

import collections



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

    response_dirs = all_dirs[all_dirs.Task == Task]
    response_dirs = response_dirs[response_dirs.Model.isin(models)].copy()
    response_dirs = response_dirs[response_dirs.Tuned.isin(tuned)]
    response_dirs = response_dirs[response_dirs.Size.isin(size)]
    print(response_dirs)
    print(response_dirs[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    # dirs = response_dirs.file
    # print (dirs)
    return response_dirs





def read_predictions(dirs_df):
    model_dict={}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        # model = row.Model + '_' +row.Size
        model = row.Model
        dir_ = join(TEST_RESULTS_PATH, dir_)
        prediction_file = [join(dir_,f) for f in listdir(dir_) if '0_testing.csv' in f][0]
        pred_df = pd.read_csv(prediction_file)
        print(pred_df.shape)
        print(pred_df.head())
        model_dict[model] = pred_df
    return model_dict

    # f = join(f, 'all_scores.csv')
    # df = pd.read_csv(f)
    # dfs.append(df)

def plot_roc(ax, y_test, y_pred_score, save_dir,color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # ax.plot(fpr, tpr, label=label + ' (area = %0.3f)' % roc_auc, linewidth=1, color=color)
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
        # average_auc = average_precision_score(y_test, y_pred_score)
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


def boxplot_csutom(data_df, ax, colors_dict, labels=None):
    means = data_df.median()
    if labels is None:
        labels = means.index
    quartile1 = data_df.quantile(0.25).reindex(means.index)
    medians = data_df.quantile(0.5).reindex(means.index)
    quartile3 = data_df.quantile(0.75).reindex(means.index)

    mins = data_df.min().reindex(means.index)
    maxs = data_df.max().reindex(means.index)

    def adjacent_values(mins, maxs, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, maxs)

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, mins, q1)
        return lower_adjacent_value, upper_adjacent_value

    whiskers = np.array([adjacent_values(mi, mx, q1, q3) for mi, mx, q1, q3 in zip(mins, maxs, quartile1, quartile3)])

    inds = np.arange(1, len(medians) + 1)
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    for x, y in zip(inds, medians):
        print(x,y)
        ax.text(x+0.2, y, "{:.2f}".format(y) , fontsize=8)

    # plot_jitter
    inds = np.arange(1, len(means) + 1)

    ax.set_xticks(inds)
    ax.set_xticklabels(labels)

    for i, s in zip(inds, means.index):
        #         ind = data[group_col] == s
        #         n = sum(ind)
        x = data_df[s]
        n = len(x)
        y = np.array([i + 0.1] * n)
        noise = np.random.normal(0, 0.01, n)
        y = y + noise
        if s in model_mapping.keys():
            model_name  = model_mapping[s]
        else:
            model_name = s
        ax.plot(y, x, '.', markersize=2, alpha=0.1, color= colors_dict[model_name])


def plot_auc_bootstrap(all_models_dict, ax, sorting_keys=None,  sort_auc=False,  pred_col = 'pred_score', metric = metrics.roc_auc_score, introduce_line_on=''):
    all_scores=[]
    names=[]
    xs=[]
    avg_scores=[]

    if sorting_keys is None:
        keys = all_models_dict.keys()
    else:
        keys = sorting_keys

    for i, k in enumerate(keys):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df[pred_col]
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metric,
                                                     n_bootstraps=1000, seed=123)
        all_scores.append(scores)
        names.append(k)
        xs.append(np.random.normal(i + 1, 0.04, len(scores)))
        avg_scores.append(score)

    if sort_auc:
        sorted_list = [(x, n) for _,x,n in sorted(zip(avg_scores, all_scores, names))]
        all_scores  = [x[0] for x in sorted_list]
        names  = [x[1] for x in sorted_list]

    all_scores_df = pd.DataFrame(all_scores).T
    labels= names
    if introduce_line_on=='(':
        labels=[n.replace(' (', '\n (') for n in names]
    elif introduce_line_on != '':
        labels=[n.replace(introduce_line_on, '\n') for n in names]
    all_scores_df.columns =names

    boxplot_csutom(all_scores_df, ax, colors_dict=model_colors, labels=labels)
    plt.ylim(0.8*all_scores_df.min().min(),1. )
    plt.locator_params(axis="y", nbins=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_auc_bootstrap_boxplot(all_models_dict, ax):
    # n = len(all_models_dict.keys())
    # colors = sns.color_palette(None, n)

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
    # (all_scores,names) = [(x, n) for _,x,n in sorted(zip(avg_scores, all_scores, names))]
    all_scores  = [x[0] for x in sorted_list]
    names  = [x[1] for x in sorted_list]
    # all_scores, names = sorted_list
    # names = [x for _, x in sorted(zip(avg_scores, names ))]

    ax.boxplot(all_scores, labels= names)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    from matplotlib import cm
    for i, (x, val, clevel, k) in enumerate(zip(xs, all_scores, clevels, names)):
        model_name = model_mapping[k]
        plt.scatter(x, val,marker='.',s=10, color=model_colors[model_name], alpha=0.05)
    plt.ylim(0.8,1.0)
    plt.locator_params(axis="y", nbins=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# def plot_auc_bootstrap(all_models_dict, ax):
#     # n = len(all_models_dict.keys())
#     # colors = sns.color_palette(None, n)
#
#     all_scores=[]
#     names=[]
#     xs=[]
#     avg_scores=[]
#     for i, k in enumerate(all_models_dict.keys()):
#         df = all_models_dict[k]
#         y_test = df['y']
#         y_pred_score = df['pred_score']
#         score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metrics.roc_auc_score,
#                                                      n_bootstraps=1000, seed=123)
#         all_scores.append(scores)
#         names.append(k)
#         xs.append(np.random.normal(i + 1, 0.04, len(scores)))
#         avg_scores.append(score)
#
#     all_scores = [x for _, x in sorted(zip(avg_scores, all_scores))]
#     names = [x for _, x in sorted(zip(avg_scores, names ))]
#
#     ax.boxplot(all_scores, labels= names)
#     ngroup = len(all_scores)
#     clevels = np.linspace(0., 1., ngroup)
#     from matplotlib import cm
#     for i, (x, val, clevel, k) in enumerate(zip(xs, all_scores, clevels, names)):
#         model_name = model_mapping[k]
#         plt.scatter(x, val,marker='.', color=model_colors[model_name], alpha=0.05)
#     plt.ylim(0.8,1.0)
#     plt.locator_params(axis="y", nbins=4)
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def plot_auc_all(all_models_dict,ax,  sorting_keys=None, sort_auc=False):
    if sorting_keys is None:
        keys = all_models_dict.keys()
    else:
        keys = sorting_keys

    print ('keys', keys)
    if sort_auc:
        keys = sort_dict(all_models_dict)

    for i, k in enumerate(keys):
        # print('model {} , auc= {}'.format(k, keys[k]))
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        print(k)
        if k in model_mapping.keys():
            model_name= model_mapping[k]
        else:
            model_name= k
        plot_roc(ax, y_test, y_pred_score, None, color=model_colors[model_name], label=model_name)


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
        plot_prc(ax, y_test, y_pred_score, None, color=model_colors[model_name], fontproperties= fontproperties, label=model_name)


fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

all_dirs= get_models()

#AUC
def plot_figure4(task='response'):
    models = ['BERT', 'CNN', 'TF-IDF']
    title = task.capitalize()
    # if task=='response':
    #     title ='Response ' + title
    # elif task =='progression':
    #     title = 'Progression ' + title
    dirs_df = filter_model_dirs(Task=task, tuned=True, models=models)
    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    dirs_df.Model = dirs_df.Model.str.replace('-NA','')

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            # row.Model = row.Model + ' (tuned on DFCI)'
            row.Model = 'DFCI-ImagingBERT'

    model_dict = read_predictions(dirs_df)

    print ('model_dict', model_dict.keys())
    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(PLOTS_PATH, 'figure4_model_arch_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename= join(saving_dir, '_auc_tuned')
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)

    ## AUC bootstrap

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax)
    filename = join(saving_dir, '_auc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)
    plt.close()

    ## PRC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_prc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(PLOTS_PATH, 'figure4_model_arch_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc_tuned')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, metric=metrics.average_precision_score)
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()



# n = 6
# current_colors = sns.color_palette(None, n)
# model_colors={}


model_colors = {'BERT': '#1f77b4',
'BERT (base)':'#1f77b4',
'BERT-base':'#1f77b4',
'BERT (med)':'#ff7f0e',
'BERT-med':'#ff7f0e',
'BERT (mini)':'#2ca02c',
'BERT-mini':'#2ca02c',
'BERT (tiny)':'#d62728',
'BERT-tiny':'#d62728',

'BERT (tuned)':'#9467bd',
'BERT-base (tuned on DFCI)':'#9467bd',
'DFCI-ImagingBERT':'#9467bd',

'BERT (original)':'#8c564b',

'clinical BERT':'#bea925',
'clinical BERT-base':'#bea925',
'Longformer':'#7f7f7f',
'Longformer (base)':'#7f7f7f',
'Longformer-base':'#7f7f7f',

'TF-IDF':'#96be25',
'CNN':'#be4d25'}

# model_names = ['BERT', 'clinical BERT', 'CNN', 'tfidf']
all_models = ['BERT', 'BERT (tuned)', 'clinical BERT', 'CNN', 'tfidf', 'Longformer']
model_mapping = {'BERT (tuned)': 'BERT (tuned)',  'BERT':'BERT', 'clinical BERT': 'clinical BERT', 'tfidf': 'TF-IDF', 'CNN': 'CNN', 'Longformer':'Longformer'}

# for i, m in enumerate(all_models):
#     name= model_mapping[m]
#     model_colors[name] = current_colors[i]

# model_colors = { 'BERT':, 'clinical BERT':, 'tfidf': , 'CNN': }

def plot_figure2_compare_domain_adaptation(task = 'response'):
    title = task.capitalize()
    models = ['BERT', 'clinical BERT']
    dirs_df = filter_model_dirs(Task=task, tuned=[True, False], models=models)

    print (dirs_df)
    print (dirs_df.shape)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]

    # sorting_keys
    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            # row.Model = row.Model + ' (tuned on DFCI)'
            row.Model = 'DFCI-ImagingBERT'
    model_dict = read_predictions(dirs_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)

    # keys = ['BERT-base', 'clinical BERT-base', 'BERT-base (tuned on DFCI)']
    keys = ['BERT-base', 'clinical BERT-base', 'DFCI-ImagingBERT']

    plot_auc_all(model_dict, ax, sorting_keys=keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(PLOTS_PATH, 'figure2_domain_adaptation_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename= join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
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
    # plt.title(title, fontsize=10)
    saving_dir = join(PLOTS_PATH, 'figure2_domain_adaptation_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metrics.average_precision_score, introduce_line_on='(')
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap')
    # plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

    metrics_dict = dict(F1=metrics.f1_score, Accuracy=metrics.accuracy_score, Precision=metrics.precision_score,
                        Recall=metrics.recall_score)
    for k, metric_func in metrics_dict.items():
        # print (k , metric_func)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
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
        filename = join(saving_dir, '_{}_confusion_matrix'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()


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
    model_dict = read_predictions(dirs_df)
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

def plot_figure1_compare_model_sizes(task='response',tuned=False ):
    # title = get_title(task, tuned)
    # title = 'Compare model size, ' + title
    title = task.capitalize()
    models = ['longformer', 'BERT']
    size = ['base', 'mini', 'med', 'tiny']
    model_mapping= { 'tfidf': 'TF-IDF', 'longformer':'Longformer'}
    dirs_df = filter_model_dirs(Task= task, tuned=tuned, models=models, size=size)
    dirs_df.Model.replace(model_mapping, inplace=True)
    # dirs_df.Model = dirs_df.Model + '_' + dirs_df.size
    # dirs_df.Model = dirs_df["Model"].astype(str) + ' ('+ dirs_df["Size"] +')'

    dirs_df.Model = dirs_df["Model"].astype(str) + '-'+ dirs_df["Size"]
    dirs_df.Model.replace({'Longformer-base': 'Longformer'}, inplace=True)
    print(dirs_df.Model)
    model_dict = read_predictions(dirs_df)

    # keys= ['Longformer (base)', 'BERT (base)', 'BERT (med)', 'BERT (mini)', 'BERT (tiny)']
    keys= ['Longformer', 'BERT-base', 'BERT-med', 'BERT-mini', 'BERT-tiny']
    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict,  ax, keys, sort_auc=False)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    saving_dir = join(PLOTS_PATH, 'figure1_model_size_{}_{}'.format(task, tuned))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename= join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)
    plt.close()



    #AUC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict,  ax, keys, sort_auc=False)
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
    saving_dir = join(PLOTS_PATH, 'figure1_model_size_{}_{}'.format(task, tuned))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax, keys, sort_auc=False, metric=metrics.average_precision_score)
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap')
    # plt.title(title, fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

    #
    metrics_dict = dict(F1=metrics.f1_score, Accuracy=metrics.accuracy_score, Precision=metrics.precision_score,Recall= metrics.recall_score)
    for k, metric_func in metrics_dict.items():
        # print (k , metric_func)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
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
        filename = join(saving_dir, '_{}_confusion_matrix'.format(k))
        plt.savefig(filename, dpi=400)
        plt.close()







plot_figure4('response')
plot_figure4('progression')
plot_figure1_compare_model_sizes(task='response',tuned=False )
plot_figure1_compare_model_sizes(task='progression',tuned=False )
plot_figure2_compare_domain_adaptation('response')
plot_figure2_compare_domain_adaptation('progression')
# plot_compare_model_arch('response')
# plot_compare_model_arch('progression')
#
# plot_compare_domain_adaptation('progression')
# plot_compare_domain_adaptation(task = 'response')
#
#
# plot_compare_seq_length(task='response', tuned=False)
# plot_compare_seq_length(task='response', tuned=True)
# plot_compare_seq_length(task='progression', tuned=True)
# plot_compare_seq_length(task='progression', tuned=False)
#


# plot_compare_model_sizes(task='response',tuned=True )
# plot_compare_model_sizes(task='progression',tuned=True )

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


