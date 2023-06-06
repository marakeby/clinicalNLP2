# __author__ = haitham elmarakeby
# from file_browser_testing import get_models
from os.path import join, exists
from config_path import TEST_RESULTS_PATH, PLOTS_PATH
from os import listdir
import pandas as pd
import numpy as np
from sklearn.metrics import *
# from utils.evaluate import evalualte
from paper_analysis_revision2.data_utils import optimal_threshold, filter_model_dirs, read_predictions
from paper_analysis_revision2.file_browser import get_models
from paper_analysis_revision2.plot_utils import cols_map, model_mapping
from utils.stat_utils import score_ci, pvalue

np.random.seed(1234)











def evaluate_ci(y_test, y_pred, y_pred_score =None):
    metric_dict= dict(
        accuracy=accuracy_score,
        precision=precision_score,
        auc= roc_auc_score,
        f1= f1_score,
        recall = recall_score,
        aupr = average_precision_score,
    matthews_corrcoef= matthews_corrcoef)

    binary_metrics = ['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef']
    avg_scores= {}
    for metric, metric_func in metric_dict.items():
        print('metric', metric)
        if metric in binary_metrics:
            y = y_pred
        else:
            y = y_pred_score
        score, ci_lower, ci_upper, scores_ = score_ci(y_test, y, score_fun=metric_func, n_bootstraps=1000, seed=123)
        formated_score= '{:.2f} [{:.2f},{:.2f}]'.format(score, ci_lower, ci_upper)
        print('metric', metric, score)
        # avg_scores.append(score)
        avg_scores[metric]= formated_score
    return avg_scores

# def compare_models(model1_pred, model2_pred):
#     y_test1, y_pred_score1 = model1_pred
#     y_test2, y_pred_score2 = model2_pred
#
#
#     score1, ci_lower1, ci_upper1, scores_1 = score_ci(y_test1, y_pred_score1, score_fun=roc_auc_score, n_bootstraps=1000, seed=123)
#     score2, ci_lower2, ci_upper2, scores_2 = score_ci(y_test2, y_pred_score2, score_fun=roc_auc_score, n_bootstraps=1000, seed=123)
#     pvalue = stats.ttest_ind(scores_1, scores_2)[1]
def save_scores(task, model_dict):

    scores = {}
    for i, k in enumerate(model_dict.keys()):
        df = model_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        y_pred = df['pred']
        if k in model_mapping.keys():
            model_name = model_mapping[k]
        else:
            model_name = k
        score = evaluate_ci(y_test, y_pred, y_pred_score =y_pred_score)
        scores[model_name] = score


    # print(scores)
    scores_df = pd.DataFrame(scores)
    print(scores_df)
    scores_df =scores_df.T
    # print (scores_df.columns)
    scores_df.columns  = [cols_map[c] for c in scores_df.columns]

    scores_df = scores_df.round(2)
    save_name = task
    if max_f1:
        save_name= save_name + '_max_f1'

    cols= ['Accuracy','Precision','AUC','F1','AUPRC','Recall','MCC',]

    rows = ['BERT_base_CNN (Frozen)',
        'BERT_med_CNN (Frozen)',
        'BERT_mini_CNN (Frozen)',
        'BERT_tiny_CNN (Frozen)',
           'longformer_base_CNN',
           'clinical BERT_base_CNN (Frozen)',
           'DFCI_ImagingBERT_base_CNN (Frozen)',
           'DFCI_ImagingBERT_base_Linear',
           'CNN_CNN',
           'TF-IDF', 'FlanT5-zeroshot_xxl']
    print(scores_df.head())
    scores_df= scores_df.loc[rows, cols]

    # find the best model in terms of AUC
    best_model = 'DFCI_ImagingBERT_base_CNN (Frozen)'

    df = model_dict[best_model]
    y_pred_score1 = df['pred_score']
    comparisons = []
    for i, k in enumerate(rows):
        print(k)
        df = model_dict[k]
        y_test = df['y']
        y_pred_score2 = df['pred_score']
        p_val = pvalue(y_test, y_pred_score1, y_pred_score2, score_fun = roc_auc_score)[0]
        comparisons.append(dict(best=best_model, compared_model=k, pvalue=p_val))
    # scores_df= scores_df[cols]
    compare_df =pd.DataFrame(comparisons)

    scores_df.to_csv(join(PLOTS_PATH, 'table_{}_ci.csv'.format(save_name)))
    compare_df.to_csv(join(PLOTS_PATH, 'compare_{}_ci.csv'.format(save_name)))

def generate_table(task):
    dirs_df = filter_model_dirs(all_dirs, Task=task, tuned=[False, True], models=models, size=size,
                                frozen=[True, False])

    def get_model_name(row):
        if row['Model'] == 'BERT' and row['Tuned']:
            row['Model']='DFCI_ImagingBERT'

        row['Model'] = '{}_{}_{}'.format(row['Model'],row['Size'], row['classifier'] )

        if ('BERT' in row['Model'] )  and row['Frozen']:
            row['Model'] = row['Model'] + ' (Frozen)'
        row['Model'] = row['Model'].replace('_NA', '')
        return row

    dirs_df = dirs_df.apply(get_model_name, axis=1)
    # for i, row in dirs_df.iterrows():
    #     if row.Tuned == True:
    #         row.Model = 'DFCI-ImagingBERT'
    #     if row['Frozen']:
    #         row['Model'] = row['Model'] + ' (Frozen)'
    # dirs_df.Model = dirs_df["Model"].astype(str) + '-' + dirs_df["Size"] + '-' + dirs_df["Task"]
    dirs_df.Model = dirs_df.Model.str.replace('-NA', '')
    model_dict = read_predictions(dirs_df, max_f1, _class=0)
    save_scores(task, model_dict)


if __name__ == '__main__':

    all_models = ['BERT', 'BERT (tuned)', 'clinical BERT', 'CNN', 'tfidf', 'Longformer', 'FlanT5-zeroshot']

    all_dirs = get_models()

    # task = 'progression'
    # task = 'response'
    max_f1 = True
    models = ['BERT', 'longformer', 'clinical BERT', 'CNN', 'TF-IDF', 'FlanT5-zeroshot']
    size = ['base', 'mini', 'med', 'tiny', 'xxl', 'NA']
    generate_table('progression')
    generate_table('response')
