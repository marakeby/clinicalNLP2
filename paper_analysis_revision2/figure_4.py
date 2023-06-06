import os
import numpy as np
from os.path import join, exists
from sklearn import metrics
from config_path import PLOTS_PATH, TEST_RESULTS_PATH
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from paper_analysis_revision2.plot_utils import plot_auc_all, plot_auc_bootstrap, plot_prc_all, plot_confusion_matrix, \
    model_mapping, cols_map, model_colors
from paper_analysis_revision2.data_utils import filter_model_dirs, read_predictions, get_all_scores
from paper_analysis_revision2.file_browser import get_models
import pandas as pd

np.random.seed(1234)

def read_files(files, max_f1=True):
    dfs = []
    for f in files:
        f = join(TEST_RESULTS_PATH, f)
        print (f)
        if 't5' in f:
            max_f1 = False
        df = get_all_scores(f, max_f1=max_f1)
        # f = join(f, 'all_scores.csv')
        # df = pd.read_csv(f)
        dfs.append(df)
    return dfs

# def filter_files(all_files, models, task='resposne', tuned=False, frozen=False, size=None ):
#     Task = task
#     # models = ['clinical BERT', 'BERT', 'longformer', 'JAMA', 'tfidf']
#     # size = ['base', 'NA']
#     # tuned = [tuned, 'NA']
#
#     if size is None:
#         size = ['base', 'NA']
#     else:
#         size= size+["NA"]
#
#     if type(tuned) == list:
#         tuned.append('NA')
#     else:
#         tuned = [tuned, 'NA']
#
#     if type(frozen) == list:
#         frozen.append('NA')
#     else:
#         frozen = [frozen, 'NA']
#
#     response_files = all_files[all_files.Task == Task]
#     response_files = response_files[response_files.Model.isin(models)].copy()
#     response_files = response_files[response_files.Tuned.isin(tuned)]
#     response_files = response_files[response_files.Size.isin(size)]
#     response_files = response_files[response_files.Frozen.isin(frozen)]
#     print(response_files)
#     print(response_files[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
#     return response_files

def plot_fig4_sizes(task='progression'):
    frozen = [True, False]
    # response_files = filter_files(all_dirs, all_models, task=task, tuned=True, frozen=[False, True])
    response_files = filter_model_dirs(all_dirs, Task=task, tuned=True, models=all_models, frozen=frozen)
    response_files = response_files.sort_values('Model', ascending=False)

    print (response_files)
    def get_model_name(row):
        if row['Model'] == 'BERT':
            if row['Tuned']:
                row['Model']='DFCI_ImagingBERT'
            row['Model'] = '{}_{}_{}'.format(row['Model'],row['Size'], row['classifier'] )
            if row['Frozen']:
                row['Model']=row['Model'] + ' (Frozen)'
        return row

    response_files = response_files.apply(get_model_name, axis=1)

    files = response_files.file.values
    models = response_files.Model.values
    # legend = models
    legend=[]
    for m in models:
        print(m)
        legend.append(m)
    saving_dir = join(PLOTS_PATH, 'figure4/{}'.format(task))
    dfs = read_files(files)
    title = task.capitalize()
    plot_sizes(dfs, legend, title,  saving_dir)

def plot_sizes(dfs, legend, title, saving_dir ):
    if not exists(saving_dir):
        os.makedirs(saving_dir)

    for  c in  cols_map.keys():
        plt.figure()
        for cc,df in zip(legend,dfs):
            print('model name', cc)
            print( df.head())
            if cc in model_mapping.keys():
                model_name = model_mapping[cc]
            else:
                model_name = cc
            x= number_patients
            y= df[c].values
            print(model_name)

            if len(y) == 1:
                plt.hlines(y=y[0], xmin=0, xmax=x[0],
                           linestyles='dotted',
                           color=model_colors[model_name])
            else:
                if 'TF-IDF' in model_name:
                    plt.plot(x, y, '--', color= model_colors[model_name])
                elif model_name=='CNN':
                    plt.plot(x, y, '-.', color= model_colors[model_name])
                else:
                    # plt.plot(x,y, '.-', color= model_colors[model_name])
                    plt.plot(x,y, '.-')

        # legend_normalized = [model_mapping[m] for m in legend]

        plt.legend(legend, loc='lower right')
        plt.xlabel('Number of patients')
        # plt.ylabel(xlabel_map[c])
        plt.ylabel(cols_map[c])
        plt.title(title)
        fname= '{}.png'.format(c)
        fname = join(saving_dir, fname)
        if c=='auc':
            plt.ylim((0.4,1.01))

        plt.savefig(fname, dpi=200)
        plt.close()

def plot_figure4(task='response'):

    title = task.capitalize()
    frozen = [True, False]
    dirs_df = filter_model_dirs(all_dirs, Task=task, tuned=True, models=all_models, frozen=frozen)
    dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"]
    dirs_df.Model = dirs_df.Model.str.replace('-NA','')

    for i, row in dirs_df.iterrows():
        print(row.Model, row.Tuned)
        if (row.Tuned == True) and (row.Model=='BERT-base'):
            row.Model = 'DFCI-ImagingBERT'
        if row.Frozen == True:
            row.Model = row.Model + ' (Frozen)'
        if row.Model=='FlanT5-zeroshot-xxl':
            row.Model = 'FlanT5-XXL (zeroshot)'

    filename = 'figure4/{}/full_data_size'.format(task)
    saving_dir = join(PLOTS_PATH, filename)

    if not exists(saving_dir):
        os.makedirs(saving_dir)
    dirs_df.to_csv(join(saving_dir, 'models.csv'))
    model_dict = read_predictions(dirs_df, max_f1=max_f1, _class=_class)

    print ('model_dict', model_dict.keys())
    ## AUC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
    # plot_auc_all(model_dict, ax, sort_auc=True)
    plot_auc_all(model_dict, ax)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)

    filename= join(saving_dir, '_auc_tuned')
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)

    ## AUC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=400)
    # plot_auc_bootstrap(model_dict, ax, introduce_line_on='(', sort_auc=True)
    plot_auc_bootstrap(model_dict, ax, introduce_line_on='(')
    filename = join(saving_dir, '_auc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel('AUROC')
    plt.savefig(filename, dpi=400)
    plt.close()

    ## PRC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
    plot_prc_all(model_dict, ax)
    # plot_prc_all(model_dict, ax, sort_auc=True)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.title(title, fontsize=10)
    # plt.xticks(fontsize=10)
    # saving_dir = join(PLOTS_PATH, 'figure4_model_arch_{}'.format(task))
    if not exists(saving_dir):
        os.mkdir(saving_dir)
    filename = join(saving_dir, '_prc_tuned')
    plt.savefig(filename, dpi=400)
    plt.close()

    # PRC bootstrap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=400)
    # plot_auc_bootstrap(model_dict, ax, metric=metrics.average_precision_score, introduce_line_on='(', sort_auc=True)
    plot_auc_bootstrap(model_dict, ax, metric=metrics.average_precision_score, introduce_line_on='(')
    ax.set_ylabel('AUPPRC')
    filename = join(saving_dir, '_prc_bootsrtap_tuned')
    plt.title(title, fontsize=10)
    plt.xticks(fontsize=8)
    plt.savefig(filename, dpi=400)
    plt.close()

if __name__ == '__main__':
    #Figure 4
    # all_models = ['BERT', 'CNN', 'TF-IDF']
    all_models = ['BERT', 'CNN', 'TF-IDF', 'FlanT5-zeroshot']
    all_dirs = get_models()
    number_patients = [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]
    # xlabel_map = dict(accuracy='Accuracy', precision='Precision', auc='AUROC', f1='F1', aupr='AUPRC', recall='Recall')

    max_f1 = True
    _class = 0

    plot_fig4_sizes('progression')
    plot_fig4_sizes('response')

    plot_figure4('response')
    plot_figure4('progression')
