from os.path import join, exists
from analysis_paper.file_browser_ import get_models
from config_path import PLOTS_PATH, GCP_RESULTS_PATH
import pandas as pd
from matplotlib import pyplot as plt
import os
base_dir = os.path.dirname(__file__)
import seaborn as sns

all_files= get_models()

number_patients= [884,592,214,103,68, 35]
cols_map=dict(accuracy='Accuracy', percision='Precision', auc='AUC', f1='F1',aupr='AUPRC', recall= 'Recall' )
xlabel_map=dict(accuracy='Accuracy', percision='Precision', auc='Area under the ROC curve (AUC)', f1='F1',aupr='Area under the precision-recall curve (AUPRC)', recall= 'Recall' )



# all_models = ['BERT', 'clinical BERT', 'JAMA', 'tfidf', 'longformer']
all_models = ['BERT',  'CNN', 'TF-IDF']
model_mapping = {'BERT': 'BERT (base, tuned, CNN head)', 'clinical BERT': 'clinical BERT', 'TF-IDF': 'TF-IDF', 'CNN': 'CNN', 'longformer':'Longformer'}
n = 5

model_colors = {'BERT': '#1f77b4',
'BERT (base)':'#1f77b4',
'BERT (base, tuned, CNN head)':'#1f77b4',
'BERT (med)':'#ff7f0e',
'BERT (mini)':'#2ca02c',
'BERT (tiny)':'#d62728',

'BERT (tuned)':'#9467bd',

'BERT (original)':'#8c564b',

'clinical BERT':'#bea925',
'Longformer':'#7f7f7f',

'TF-IDF':'#96be25',
'CNN':'#be4d25'}

# current_colors = sns.color_palette(None, n)
# model_colors={}
# for i, m in enumerate(all_models):
#     # name= model_mapping[m]
#     model_colors[m] = current_colors[i]

def read_files(files):
    dfs = []
    for f in files:
        f = join(GCP_RESULTS_PATH, f)
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
            if cc in model_mapping.keys():
                model_name = model_mapping[cc]
            else:
                model_name = cc
            x= number_patients
            y= df[c].values
            print(model_name)
            if 'TF-IDF' in model_name:
                plt.plot(x, y, '--', color= model_colors[model_name])
            elif model_name=='CNN':
                plt.plot(x, y, '-.', color= model_colors[model_name])
            else:
                plt.plot(x,y, '.-', color= model_colors[model_name])
        legend_normalized = [model_mapping[m] for m in legend]

        plt.legend(legend_normalized)
        plt.xlabel('number of patients')
        plt.ylabel(xlabel_map[c])
        plt.title(title)
        fname= '{}.png'.format(c)
        fname = join(saving_dir, fname)
        if c=='auc':
            plt.ylim((0.5,1.01))
        plt.savefig(fname, dpi=200)
        plt.close()


def filter_files(all_files, models, task='resposne', tuned=False ):
    Task = task
    # models = ['clinical BERT', 'BERT', 'longformer', 'JAMA', 'tfidf']
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
    response_files = filter_files(all_files, all_models, task='response', tuned=False)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(PLOTS_PATH, 'response_originals')
    dfs = read_files(files)
    title = 'Response (original BERT)'
    plot_files(dfs, legend, title, saving_dir)


def plot_tuned():
    response_files = filter_files(all_files, all_models, task='response', tuned=True)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(PLOTS_PATH, 'response_tuned')
    dfs = read_files(files)
    title = 'Response, increasing sample sizes'
    plot_files(dfs, legend, title, saving_dir)

def plot_originals_progression():
    response_files = filter_files(all_files, all_models, task='progression', tuned=False)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(PLOTS_PATH, 'progression_originals')
    dfs = read_files(files)
    title = 'Progression (original BERT)'
    plot_files(dfs, legend,title,  saving_dir)

def plot_tuned_progression():
    response_files = filter_files(all_files, all_models, task='progression', tuned=True)
    files = response_files.file.values
    models = response_files.Model.values
    legend = models
    saving_dir = join(PLOTS_PATH, 'progression_tuned')
    dfs = read_files(files)
    title = 'Progression, increasing sample sizes'
    plot_files(dfs, legend, title,  saving_dir)

# plot_originals()
plot_tuned()
#
# plot_originals_progression()
plot_tuned_progression()



