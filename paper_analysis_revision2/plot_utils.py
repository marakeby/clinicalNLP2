# __author__ = haitham elmarakeby

from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.stat_utils import score_ci
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
import itertools
import collections
from matplotlib.ticker import FormatStrFormatter



fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
cols_map=dict(accuracy='Accuracy', precision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' ,  matthews_corrcoef='MCC')
model_mapping = {'BERT (tuned)': 'DFCI-ImagingBERT',  'BERT':'BERT', 'clinical BERT': 'clinical BERT', 'tfidf': 'TF-IDF', 'CNN': 'CNN', 'Longformer':'Longformer',
                 'FlanT5-zeroshot-xxl': 'FlanT5-XXL (zero-shot)'}
model_colors = {'BERT': '#1f77b4',
'BERT (base)':'#1f77b4',
'BERT-base':'#8c564b',
'BERT (med)':'#9467bd',
'BERT-med\n(Frozen)':'#9467bd',
'BERT-med (Frozen)':'#9467bd',
'BERT-med':'#9467bd',
'BERT (mini)':'#2ca02c',
'BERT-mini\n(Frozen)':'#2ca02c',
'BERT-mini (Frozen)':'#2ca02c',
'BERT-mini':'#2ca02c',
'BERT (tiny)':'#d62728',
'BERT-tiny\n(Frozen)':'#d62728',
'BERT-tiny (Frozen)':'#d62728',
'BERT-tiny':'#d62728',

'BERT (tuned)':'#9467bd',
'BERT-base (tuned on DFCI)':'#9467bd',
'DFCI-ImagingBERT':'#ff7f0e',
'DFCI-ImagingBERT (Frozen)':'#1f77b4',
'DFCI-ImagingBERT\n(Frozen)':'#1f77b4',

'BERT (original)':'#8c564b',
'BERT-base (Frozen)':'#8c564b',
'BERT-base(Frozen)':'#8c564b',

'BERT-base\n(Frozen)':'#8c564b',

'clinical BERT':'#bea925',
'clinical BERT-base':'#bea925',
'clinical BERT-base (Frozen)':'#bea925',
'Longformer':'#7f7f7f',
'Longformer\n(Frozen)':'#7f7f7f',
'Longformer (Frozen)':'#7f7f7f',
'Longformer (base)':'#7f7f7f',
'Longformer-base':'#7f7f7f',

'TF-IDF':'#96be25',
'CNN':'#be4d25',
'FlanT5-zeroshot-xxl':'#008080',
'FlanT5-XXL (zeroshot)':'#008080',
'FlanT5-zeroshot':'#008080',
                }

def plot_prc(ax, y_test, y_pred_score, color, fontproperties, label=''):
    # plt.figure(fig.number)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score)
    roc_auc = average_precision_score(y_test, y_pred_score)
    ax.plot(recall, precision, label=label + '(area= %0.2f)' % roc_auc, linewidth=2, color=color)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall',fontproperties )
    ax.set_ylabel('Precision', fontproperties)

def plot_roc(ax, y_test, y_pred_score, color, fontproperties, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=1.5, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)


def plot_confusion_matrix(ax, cm, classes, fontproperties, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fontsize= fontproperties['size']
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label',  fontproperties )
    ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes,  fontdict = fontproperties)
    ax.set_xticklabels(classes)
    ax.set_yticks([t-0.25 for t in tick_marks])
    ax.set_yticklabels(classes, fontdict = fontproperties, rotation=90)
    ax.set_yticklabels(classes)


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
        plot_roc(ax, y_test, y_pred_score, model_colors[model_name], fontproperties, model_name)


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
        plot_prc(ax, y_test, y_pred_score, color=model_colors[model_name], fontproperties= fontproperties, label=model_name)

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
        # if k in model_mapping.keys():
        #     k= model_mapping[k]
        # else:
        #     model_name= k

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


def plot_confusion_matrix(ax, cm, classes, fontproperties, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fontsize= fontproperties['size']
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label',  fontproperties )
    ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes,  fontdict = fontproperties)
    ax.set_xticklabels(classes)
    ax.set_yticks([t-0.25 for t in tick_marks])
    ax.set_yticklabels(classes, fontdict = fontproperties, rotation=90)
    ax.set_yticklabels(classes)