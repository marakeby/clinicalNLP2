from os.path import join, exists
from analysis.file_browser_ import get_models
import pandas as pd
from matplotlib import pyplot as plt
import os
from os import listdir

from analysis.stat_utils import score_ci

base_dir = os.path.dirname(__file__)
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
        dir_ = join(base_dir, dir_)
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
        symbol = '-'
    elif 'JAMA' in label:
        symbol = '-'
    ax.plot(fpr, tpr, symbol, label=label + ' (%0.3f)' % roc_auc, linewidth=1, color=color)

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


def plot_auc_bootstrap(all_models_dict, ax):
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

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

    all_scores = [x for _, x in sorted(zip(avg_scores, all_scores))]
    names = [x for _, x in sorted(zip(avg_scores, names ))]

    ax.boxplot(all_scores, labels= names)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    from matplotlib import cm
    for i, (x, val, clevel) in enumerate(zip(xs, all_scores, clevels)):
        plt.scatter(x, val,marker='.', color=colors[i], alpha=0.1)


def plot_auc_all(all_models_dict, ax):
    # sort based on area under prc
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

    sorted_dict = sort_dict(all_models_dict)
    for i, k in enumerate(sorted_dict.keys()):
        print('model {} , auc= {}'.format(k, sorted_dict[k]))
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        plot_roc(ax, y_test, y_pred_score, None, color=colors[i], label=k)


fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

all_dirs= get_models()

def plot_compare_model_arch(task='response'):
    models = ['BERT', 'JAMA', 'tfidf']
    title = 'Compare Model Arch. (Original BERT)'
    if task=='response':
        title ='Response, ' + title
    elif task =='progression':
        title = 'Progression, ' + title
    dirs_df = filter_model_dirs(Task=task, tuned=False, models=models)
    model_dict = read_predictions(dirs_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(base_dir, 'plots2/model_arch_{}'.format(task))
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


def plot_compare_domain_adaptation(task = 'response'):
    title= 'Domain adaptation (Tuned BERT)'
    if task=='response':
        title ='Response, ' + title
    elif task =='progression':
        title = 'Progression, ' + title

    models = ['BERT', 'clinical BERT', 'JAMA', 'tfidf']
    model_mapping= { 'BERT':'BERT (tuned)', 'clinical BERT':'clinical BERT', 'tfidf': 'TF-IDF', 'JAMA': 'JAMA'}
    dirs_df = filter_model_dirs(Task=task, tuned=True, models=models)
    dirs_df.Model.replace(model_mapping, inplace=True)
    print(dirs_df.Model)
    model_dict = read_predictions(dirs_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(base_dir, 'plots2/domain_adaptation_{}'.format(task))
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
    saving_dir = join(base_dir, 'plots2/seq_length_{}_{}'.format(task, tuned))
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

def plot_compare_model_sizes(task='response',tuned=False ):
    title = get_title(task, tuned)
    title = 'Compare model size, ' + title
    models = ['BERT']
    size = ['base', 'mini', 'med', 'tiny']
    model_mapping= { 'tfidf': 'TF-IDF'}
    dirs_df = filter_model_dirs(Task= task, tuned=tuned, models=models, size=size)
    dirs_df.Model.replace(model_mapping, inplace=True)
    print(dirs_df.Model)
    model_dict = read_predictions(dirs_df)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    saving_dir = join(base_dir, 'plots2/model_size_{}_{}'.format(task, tuned))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename= join(saving_dir, '_auc')
    plt.savefig(filename, dpi=400)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_auc_bootstrap(model_dict, ax)
    filename = join(saving_dir, '_auc_bootsrtap')
    plt.title(title , fontsize=10)
    plt.savefig(filename, dpi=400)
    plt.close()

plot_compare_model_arch('response')
plot_compare_model_arch('progression')
#
plot_compare_domain_adaptation('progression')
plot_compare_domain_adaptation(task = 'response')


plot_compare_seq_length(task='response', tuned=False)
plot_compare_seq_length(task='response', tuned=True)
plot_compare_seq_length(task='progression', tuned=True)
plot_compare_seq_length(task='progression', tuned=False)

plot_compare_model_sizes(task='response',tuned=False )
plot_compare_model_sizes(task='response',tuned=True )
plot_compare_model_sizes(task='progression',tuned=False )
plot_compare_model_sizes(task='progression',tuned=True )

'''
Model architecture
	BERT, JAMA, TF-IDF
Domain adaptation
	BERT_tuned,  clinical BERT, JAMA, TF-IDF
Sequence Length 
	BERT,  Longformer , clinical BERT, JAMA, TF-IDF
	BERT_tuned,  Longformer _tuned, clinical BERT, JAMA, TF-IDF
Model size
	base, tiny, med, mini
	base, tiny, med, mini (tuned)
Training sizes
'''


