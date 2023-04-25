from os.path import join, exists
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix, precision_recall_curve
from paper_analysis.plot_utils import plot_confusion_matrix
from file_browser_testing import get_models
import pandas as pd
from matplotlib import pyplot as plt, gridspec
import os
from os import listdir
from utils.stat_utils import score_ci
from config_path import TEST_RESULTS_PATH, PLOTS_PATH
import numpy as np
from sklearn import metrics
import collections
from scipy.ndimage.filters import gaussian_filter1d

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

all_models = ['BERT', 'BERT (tuned)', 'clinical BERT', 'CNN', 'tfidf', 'Longformer']
model_mapping = {'BERT (tuned)': 'BERT (tuned)',  'BERT':'BERT', 'clinical BERT': 'clinical BERT', 'tfidf': 'TF-IDF', 'CNN': 'CNN', 'Longformer':'Longformer'}

fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

all_dirs= get_models()




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
    response_dirs = all_dirs[all_dirs.Task == Task]
    response_dirs = response_dirs[response_dirs.Model.isin(models)].copy()
    response_dirs = response_dirs[response_dirs.Tuned.isin(tuned)]
    response_dirs = response_dirs[response_dirs.Size.isin(size)]
    print(response_dirs)
    print(response_dirs[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    return response_dirs


def read_predictions(dirs_df, _class= 0):
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

        if max_f1:
            prediction_file_train = [join(dir_, f) for f in listdir(dir_) if '0_traing.csv' in f][0]
            pred_df_train = pd.read_csv(prediction_file_train)

            if _class == 1:
                pred_df_train['pred_score'] = 1 - pred_df_train['pred_score']
                pred_df_train['pred'] = 1 - pred_df_train['pred']
                pred_df_train['y'] = 1 - pred_df_train['y']

            y_pred_score_train = pred_df_train['pred_score']
            # y_pred = pred_df_train['pred']
            # y_train = pred_df_train['y']
            # precision, recall, thresholds = precision_recall_curve(y_train, y_pred_score)
            # f1_scores = 2 * recall * precision / (recall + precision)
            # th = thresholds[np.argmax(f1_scores)]

            th, f1s = optimal_threshold(y_pred_score_train)
            y_pred = pred_df['pred_score'] >= th
            pred_df['pred'] = y_pred

            # print(k)
            # print('Best threshold: ', thresholds[np.argmax(f1_scores)])
            # print('Best F1-Score: ', np.max(f1_scores))
            # pred_df['pred'] = pred_df['pred_score'] >th
        # else:
        #     ext=''
        #     y_pred = df['pred']

        # print(pred_df.shape)
        # print(pred_df.head())
        model_dict[model] = pred_df
        # model_dict[model+'_train'] = pred_df_train
    return model_dict

def plot_roc(ax, y_test, y_pred_score, save_dir,color, label=''):
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
    ax.set_ylim(0.8*all_scores_df.min().min(),1. )
    ax.locator_params(axis="y", nbins=4)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)



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
        plot_roc(ax, y_test, y_pred_score, None, color=model_colors[model_name], label=model_name)

    ax.legend(loc="lower right", fontsize=6, framealpha=0.0)



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




def get_results_figure_1(task, tuned):
    models = ['longformer', 'BERT']
    size = ['base', 'mini', 'med', 'tiny']
    model_mapping= { 'tfidf': 'TF-IDF', 'longformer':'Longformer'}
    dirs_df = filter_model_dirs(Task= task, tuned=tuned, models=models, size=size)
    dirs_df.Model.replace(model_mapping, inplace=True)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-'+ dirs_df["Size"]
    dirs_df.Model.replace({'Longformer-base': 'Longformer'}, inplace=True)
    print(dirs_df.Model)
    model_dict = read_predictions(dirs_df, _class=_class)
    return model_dict


def get_results_figure2(task):
    models = ['BERT', 'clinical BERT']

    dirs_df = filter_model_dirs(Task=task, tuned=[True, False], models=models)

    print(dirs_df)
    print(dirs_df.shape)

    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]

    # sorting_keys
    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            # row.Model = row.Model + ' (tuned on DFCI)'
            row.Model = 'DFCI-ImagingBERT'
    model_dict = read_predictions(dirs_df, _class=_class)
    return model_dict


def get_results_figure_4(task):
    models = ['BERT', 'CNN', 'TF-IDF']
    dirs_df = filter_model_dirs(Task=task, tuned=True, models=models)
    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    dirs_df.Model = dirs_df.Model.str.replace('-NA', '')

    for i, row in dirs_df.iterrows():
        if row.Tuned == True:
            row.Model = 'DFCI-ImagingBERT'

    model_dict = read_predictions(dirs_df, _class=_class)
    return model_dict



def plot_figure_1():
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5, 4), dpi=200)

    fig = plt.figure(constrained_layout=False, figsize=(8, 7))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, width_ratios=[20, 2, 20], height_ratios=[20, 2, 20])

    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 2])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[2, 2])

    keys = ['Longformer', 'BERT-base', 'BERT-med', 'BERT-mini', 'BERT-tiny']
    task  = 'progression'
    model_dict = get_results_figure_1(task= task, tuned=False)

    plot_auc_all(model_dict, ax1, keys, sort_auc=False)
    ax1.legend(loc="lower right", fontsize=6, framealpha=0.0)
    plot_auc_bootstrap(model_dict, ax2, sorting_keys=keys, sort_auc=False, introduce_line_on='(')
    # plot_auc_bootstrap(model_dict, ax2, keys, sort_auc=False)
    ax2.set_ylabel('AUROC')
    ax2.tick_params(axis='x', which='major', labelsize=7)


    ax1.set_title(task.capitalize())
    ax2.set_title(task.capitalize())

    #####------------
    task = 'response'
    model_dict = get_results_figure_1(task=task, tuned=False)

    plot_auc_all(model_dict, ax3, keys, sort_auc=False)
    plot_auc_bootstrap(model_dict, ax4, keys, sort_auc=False)
    ax4.tick_params(axis='x', which='major', labelsize=7)

    ax4.set_ylabel('AUROC')
    ax3.set_title(task.capitalize())
    ax4.set_title(task.capitalize())


    saving_dir = join(PLOTS_PATH, 'figure1')
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, 'figure1_auc.pdf')
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_figure_2():
    fig = plt.figure(constrained_layout=False, figsize=(8, 7))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, width_ratios=[20, 2, 20], height_ratios=[20, 2, 20])

    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 2])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[2, 2])
    keys = ['BERT-base', 'clinical BERT-base', 'DFCI-ImagingBERT']
    task  = 'progression'
    model_dict = get_results_figure2(task= task)

    plot_auc_all(model_dict, ax1, sorting_keys=keys, sort_auc=False)

    plot_auc_bootstrap(model_dict, ax2, sorting_keys=keys, sort_auc=False, introduce_line_on='(')
    ax2.tick_params(axis='x', which='major', labelsize=7)
    ax2.set_ylabel('AUROC')
    ax1.set_title(task.capitalize())
    ax2.set_title(task.capitalize())

    #####------------
    task = 'response'
    model_dict = get_results_figure2(task=task)

    plot_auc_all(model_dict, ax3, sorting_keys=keys, sort_auc=False)

    plot_auc_bootstrap(model_dict, ax4, sorting_keys=keys, sort_auc=False, introduce_line_on='(')
    ax4.tick_params(axis='x', which='major', labelsize=7)
    ax4.set_ylabel('AUROC')
    ax3.set_title(task.capitalize())
    ax4.set_title(task.capitalize())

    saving_dir = join(PLOTS_PATH, 'figure2')
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, 'figure2_auc.pdf')
    plt.savefig(filename, dpi=200)
    plt.close()


def read_files(files):
    dfs = []
    for f in files:
        f = join(TEST_RESULTS_PATH, f)
        f = join(f, 'all_scores.csv')
        df = pd.read_csv(f)
        dfs.append(df)
    return dfs

def get_results_figure_4_sizes(task='response'):
    models = ['BERT',  'CNN', 'TF-IDF']
    response_files = filter_model_dirs( models=models,Task=task, tuned=True)
    files = response_files.file.values
    dfs = read_files(files)
    legend = response_files.Model.values
    return dfs, legend


def plot_sizes(dfs, legend, ax, metric='auc' ):
    model_mapping = {'BERT': 'DFCI-ImagingBERT',  'TF-IDF': 'TF-IDF', 'CNN': 'CNN',
                     'longformer': 'Longformer'}

    for cc,df in zip(legend,dfs):
        if cc in model_mapping.keys():
            model_name = model_mapping[cc]
        else:
            model_name = cc
        x= number_patients
        y= df[metric].values
        print(model_name)

        if 'TF-IDF' in model_name:
            ax.plot(x, y, '--', color= model_colors[model_name])
        elif model_name=='CNN':
            ax.plot(x, y, '-.', color= model_colors[model_name])
        else:
            ax.plot(x,y, '.-', color= model_colors[model_name])
        legend_normalized = [model_mapping[m] for m in legend]

        ax.legend(legend_normalized, loc='lower right')
        ax.set_xlabel('Number of patients')
        ax.set_ylabel(cols_map[metric])


def plot_figure_4():

    fig = plt.figure(constrained_layout=False, figsize=(8, 7))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, width_ratios=[20, 2, 20], height_ratios=[20, 2, 20])

    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 2])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[2, 2])
    task = 'progression'
    dfs, legend = get_results_figure_4_sizes(task)

    plot_sizes(dfs, ax=ax1, legend=legend, metric = 'auc')

    model_dict = get_results_figure_4(task)
    plot_auc_bootstrap(model_dict, ax2)
    ax2.set_ylabel('AUROC')
    ax1.set_title(task.capitalize())
    ax2.set_title(task.capitalize())

    task = 'response'
    dfs, legend = get_results_figure_4_sizes(task)

    plot_sizes(dfs, ax=ax3, legend=legend, metric='auc')

    model_dict = get_results_figure_4(task)

    plot_auc_bootstrap(model_dict, ax4)
    ax4.set_ylabel('AUROC')
    ax3.set_title(task.capitalize())
    ax4.set_title(task.capitalize())

    saving_dir = join(PLOTS_PATH, 'figure4')
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, 'figure4_auc.pdf')
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_figure_3_sub(task, ax):
    base_dir = ''
    base_dir = join(TEST_RESULTS_PATH, base_dir)
    log_dir = join(base_dir, 'bert_classifier')
    if task == 'progression':
        log_dir_ = join(log_dir, 'progression_one_split_BERT_sizes_base_frozen_tuned_Nov-15_15-36')
        title = 'Progression'
    elif task == 'response':
        log_dir_ = join(log_dir, 'response_one_split_BERT_sizes_base_frozen_tuned_Nov-16_02-49')
        title = 'Response'
    else:
        raise

    filename = join(log_dir_, 'all_scores.csv')
    df = pd.read_csv(filename, index_col=0)

    cnn_ind = df.index.str.contains('CNN')
    rnn_ind = df.index.str.contains('RNN')
    linear_ind = df.index.str.contains('Linear')

    cnn_df = df[cnn_ind].copy()
    rnn_df = df[rnn_ind].copy()
    linear_df = df[linear_ind].copy()

    dfs = [cnn_df, rnn_df, linear_df]
    legend = ['CNN', 'RNN', 'Linear']


    number_patients = [884, 592, 214, 103, 68, 35]

    for df in dfs:
        x = number_patients
        y = df['auc'].values

        ysmoothed = gaussian_filter1d(y, sigma=0.5)

        # ax.plot(x, y, '.-')
        ax.plot(x, ysmoothed, '.-')
    ax.legend(legend, loc='lower right')
    ax.set_xlabel('Number of patients')
    ax.set_ylabel('AUROC')
    ax.set_ylim((0, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_figure_3():
    fig = plt.figure(constrained_layout=False, figsize=(8, 4))
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, width_ratios=[20, 2, 20])

    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 2])

    task = 'progression'
    plot_figure_3_sub(task, ax1)
    ax1.set_title(task.capitalize())

    task = 'response'
    plot_figure_3_sub(task, ax2)
    ax2.set_title(task.capitalize())
    saving_dir = join(PLOTS_PATH, 'figure3')
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, 'figure3_auc.pdf')
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == '__main__':
    #Figure 1
    max_f1 = True
    _class =  0 ## _class=1 flips the prediction scores and labels (default = 0)

    plot_figure_1()
    plot_figure_2()
    plot_figure_4()
    plot_figure_3()



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


