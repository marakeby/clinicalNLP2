from os.path import join, exists
from analysis.file_browser_ import get_models

import pandas as pd
from matplotlib import pyplot as plt
import os
base_dir = os.path.dirname(__file__)

all_files= get_models()

number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )

def read_files(files):
    dfs = []
    for f in files:
        f = join(base_dir, f)
        f = join(f, 'all_scores.csv')
        df = pd.read_csv(f)
        dfs.append(df)
    return dfs

def plot_files(dfs, legend, title, saving_dir ):
    if not exists(saving_dir):
        os.mkdir(saving_dir)

    for  c in  cols_map.keys():
        plt.figure()
        for model_name,df in zip(legend,dfs):
            x= number_patients
            y= df[c].values
            print(model_name)
            if 'tfidf' in model_name:
                plt.plot(x, y, '--')
            elif 'JAMA' in model_name:
                plt.plot(x, y, '-.')
            else:
                plt.plot(x,y, '.-')
        plt.legend(legend)
        plt.xlabel('number of patients')
        plt.ylabel(cols_map[c])
        plt.title(title)
        fname= '{}.png'.format(c)
        fname = join(saving_dir, fname)
        plt.savefig(fname)
        plt.close()


def filter_files(all_files, task='resposne', tuned=False ):
    Task = task
    models = ['clinical BERT', 'BERT', 'longformer', 'JAMA', 'tfidf']
    size = ['base', 'NA']
    tuned = [tuned, 'NA']
    response_files = all_files[all_files.Task == Task]
    response_files = response_files[response_files.Model.isin(models)].copy()
    response_files = response_files[response_files.Tuned.isin(tuned)]
    response_files = response_files[response_files.Size.isin(size)]
    print(response_files)
    print(response_files[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    return response_files

def plot_originals():
    response_files = filter_files(all_files, task='response', tuned=False)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(base_dir, 'plots2/response_originals')
    dfs = read_files(files)
    title = 'Response (original BERT)'
    plot_files(dfs, legend, title, saving_dir)


def plot_tuned():
    response_files = filter_files(all_files, task='response', tuned=True)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(base_dir, 'plots2/response_tuned')
    dfs = read_files(files)
    title = 'Response (tuned BERT)'
    plot_files(dfs, legend, title, saving_dir)

def plot_originals_progression():
    response_files = filter_files(all_files, task='progression', tuned=False)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(base_dir, 'plots2/progression_originals')
    dfs = read_files(files)
    title = 'Progression (original BERT)'
    plot_files(dfs, legend,title,  saving_dir)

def plot_tuned_progression():
    response_files = filter_files(all_files, task='progression', tuned=True)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(base_dir, 'plots2/progression_tuned')
    dfs = read_files(files)
    title = 'Progression (Tuned BERT)'
    plot_files(dfs, legend, title,  saving_dir)

plot_originals()
plot_tuned()
#
plot_originals_progression()
plot_tuned_progression()



