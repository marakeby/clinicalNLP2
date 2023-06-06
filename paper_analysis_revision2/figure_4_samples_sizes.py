from os.path import join, exists
from file_browser import get_models
from config_path import PLOTS_PATH, TEST_RESULTS_PATH
import pandas as pd
from matplotlib import pyplot as plt
import os
base_dir = os.path.dirname(__file__)

all_files= get_models()

# number_patients= [884,592,214,103,68, 35]
# number_patients= [884,592,214,103,68, 35]
number_patients= [884, 700, 500, 300, 200, 100, 70, 50, 30, 10]

cols_map=dict(accuracy='Accuracy', precision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )
xlabel_map=dict(accuracy='Accuracy', precision='Precision', auc='AUROC', f1='F1',aupr='AUPRC', recall= 'Recall' )

all_models = ['BERT',  'CNN', 'TF-IDF']
# model_mapping = {'BERT': 'DFCI-ImagingBERT', 'clinical BERT': 'clinical BERT', 'TF-IDF': 'TF-IDF', 'CNN': 'CNN', 'longformer':'Longformer'}
model_mapping = {'BERT': 'BERT', 'clinical BERT': 'clinical BERT', 'TF-IDF': 'TF-IDF', 'CNN': 'CNN', 'longformer':'Longformer'}
n = 5

model_colors = {'BERT': '#1f77b4',
'BERT (base)':'#1f77b4',
'BERT (base, tuned, CNN head)':'#1f77b4',
'BERT-base (tuned on DFCI)':'#1f77b4',
'DFCI-ImagingBERT':'#1f77b4',
'BERT (med)':'#ff7f0e',
'BERT (mini)':'#2ca02c',
'BERT (tiny)':'#d62728',

'BERT (tuned)':'#9467bd',

'BERT (original)':'#8c564b',

'clinical BERT':'#bea925',
'Longformer':'#7f7f7f',

'TF-IDF':'#96be25',
'CNN':'#be4d25'}

def read_files(files):
    dfs = []
    for f in files:
        f = join(TEST_RESULTS_PATH, f)
        f = join(f, 'all_scores.csv')
        df = pd.read_csv(f)
        dfs.append(df)
    return dfs

def plot_files(dfs, legend, title, saving_dir ):
    if not exists(saving_dir):
        os.mkdir(saving_dir)

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
            # print(model_name)

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
        plt.ylabel(xlabel_map[c])
        plt.title(title)
        fname= '{}.png'.format(c)
        fname = join(saving_dir, fname)
        if c=='auc':
            plt.ylim((0.4,1.01))

        plt.savefig(fname, dpi=200)
        plt.close()


def filter_files(all_files, models, task='resposne', tuned=False, frozen=False, size=None ):
    Task = task
    # models = ['clinical BERT', 'BERT', 'longformer', 'JAMA', 'tfidf']
    # size = ['base', 'NA']
    # tuned = [tuned, 'NA']

    if size is None:
        size = ['base', 'NA']
    else:
        size= size+["NA"]

    if type(tuned) == list:
        tuned.append('NA')
    else:
        tuned = [tuned, 'NA']

    if type(frozen) == list:
        frozen.append('NA')
    else:
        frozen = [frozen, 'NA']

    response_files = all_files[all_files.Task == Task]
    response_files = response_files[response_files.Model.isin(models)].copy()
    response_files = response_files[response_files.Tuned.isin(tuned)]
    response_files = response_files[response_files.Size.isin(size)]
    response_files = response_files[response_files.Frozen.isin(frozen)]
    print(response_files)
    print(response_files[['Frozen', 'Model', 'Size', 'Task', 'Tuned', 'classifier']])
    return response_files

def plot_originals():
    response_files = filter_files(all_files, all_models, task='response', tuned=False, frozen=False)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models

    saving_dir = join(PLOTS_PATH, 'response_originals')
    dfs = read_files(files)
    title = 'Response (original BERT)'
    plot_files(dfs, legend, title, saving_dir)


def plot_tuned():
    response_files = filter_files(all_files, all_models, task='response', tuned=True, frozen=False)
    response_files = response_files.sort_values('Model', ascending=False)

    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    print('legend', legend)
    saving_dir = join(PLOTS_PATH, 'figure4_response_tuned_sample_sizes')
    dfs = read_files(files)
    title = 'Response'
    plot_files(dfs, legend, title, saving_dir)

def plot_originals_progression():
    response_files = filter_files(all_files, all_models, task='progression', tuned=[True, False], frozen=[True, False])
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(PLOTS_PATH, 'progression_originals')
    dfs = read_files(files)
    title = 'Progression (original BERT)'
    plot_files(dfs, legend,title,  saving_dir)

def plot_tuned_progression():
    response_files = filter_files(all_files, all_models, task='progression', tuned=True, frozen=[False, True])
    response_files = response_files.sort_values('Model', ascending=False)

    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    filename= 'figure4_progression_tuned_sample_sizes'
    saving_dir = join(PLOTS_PATH, filename)
    response_files.to_csv(join(saving_dir, filename+'.csv'))
    dfs = read_files(files)
    title = 'Progression'
    plot_files(dfs, legend, title,  saving_dir)

def plot_models(task='progression'):
    response_files = filter_files(all_files, all_models, task=task, tuned=True, frozen=[False, True])
    response_files = response_files.sort_values('Model', ascending=False)

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
    saving_dir = join(PLOTS_PATH, 'figure4_{}'.format(task))
    dfs = read_files(files)
    title = task.capitalize()
    plot_files(dfs, legend, title,  saving_dir)


def plot_arch_size():
    all_models = ['BERT']
    response_files = filter_files(all_files, all_models, task='progression', size=['base', 'med', 'mini', 'tiny'], tuned=False, frozen=[ False])
    response_files = response_files.sort_values('Model', ascending=False)

    def get_model_name(row):
        if row['Model'] == 'BERT':
            if row['Tuned']:
                row['Model'] = 'DFCI_ImagingBERT'
            row['Model'] = '{}_{}_{}'.format(row['Model'], row['Size'], row['classifier'])
            if row['Frozen']:
                row['Model'] = row['Model'] + ' (Frozen)'
        return row

    response_files = response_files.apply(get_model_name, axis=1)

    files = response_files.file.values
    models = response_files.Model.values
    # legend = models
    legend = []
    for m in models:
        print(m)
        legend.append(m)
    saving_dir = join(PLOTS_PATH, 'arch_size')
    dfs = read_files(files)
    title = 'Progression (arch size)'
    plot_files(dfs, legend, title, saving_dir)

def plot_clinical(task='progression'):
    models = ['clinical BERT', 'CNN', 'TF-IDF']
    response_files = filter_files(all_files, models, task =task , size=['base'], tuned=[False, True], frozen=[ False, True])
    response_files = response_files.sort_values('Model', ascending=False)

    def get_model_name(row):
        if 'BERT' in row['Model'] :
            if row['Tuned']==True:
                row['Model'] = 'DFCI_ImagingBERT'
            row['Model'] = '{}_{}_{}'.format(row['Model'], row['Size'], row['classifier'])
            if row['Frozen']:
                row['Model'] = row['Model'] + ' (Frozen)'
        return row

    response_files = response_files.apply(get_model_name, axis=1)

    files = response_files.file.values
    models = response_files.Model.values
    # legend = models
    legend = []
    for m in models:
        print(m)
        legend.append(m)
    saving_dir = join(PLOTS_PATH, 'clinical_{}'.format(task))
    dfs = read_files(files)
    title = '{} (clinical)'.format(task)
    plot_files(dfs, legend, title, saving_dir)



if __name__ == '__main__':
    # plot_originals()
    # plot_tuned()

    # plot_originals_progression()
    # plot_tuned_progression()
    plot_models('progression')
    plot_models('response')
    # plot_arch_size()
    # plot_clinical()
    # plot_clinical(task='response')

