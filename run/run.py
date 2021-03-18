
random_seed = 234
from utils.rnd import set_random_seeds
set_random_seeds(random_seed = random_seed)

from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from pipeline.one_split import OneSplitPipeline
from utils.logs import set_logging
import logging
import datetime
import imp
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
params_files = []

import torch
torch.manual_seed(1234)
import numpy as np
np.random.seed(1234)
from os.path import join, dirname, realpath
current_dir = dirname(realpath(__file__))
sys.path.insert(0, dirname(current_dir))

from config_path import PROSTATE_LOG_PATH
#any_cancer
# params_files.append('./params/manual_labels/any_cancer_one_split_tfidf')
# params_files.append('./params/manual_labels/any_cancer_crossvalidation_tfidf')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_impress')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_impress_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_impress_150')


# params_files.append('./params/manual_labels/any_cancer_one_split_seq_500')
# params_files.append('./params/baseline_size/any_cancer_one_split_seq')
# params_files.append('./params/baseline_size/progression_one_split_seq')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_NARR_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_impress_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_impress_150')

# params_files.append('./params/manual_labels/response_one_split_seq')
# params_files.append('./params/manual_labels/response_one_split_seq_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq_150')
# params_files.append('./params/manual_labels/any_cancer_one_split_seq')
# params_files.append('./params/manual_labels/response_one_split_seq_500')
# params_files.append('./params/manual_labels/progression_one_split_seq_500')
# params_files.append('./params/manual_labels/any_cancer_one_split_tfidf')
# params_files.append('./params/bert/any_cancer_one_split_seq_IMPRESS')
# params_files.append('./params/bert/any_cancer_one_split_BERT_IMPRESS')
# params_files.append('./params/bert/any_cancer_one_split_BERT_IMPRESS_NARR')
# params_files.append('./params/bert/any_cancer_one_split_BERT_IMPRESS_NARR')
# params_files.append('./params/bert/any_cancer_one_split_BERT_rnn')
# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn')
# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn_tiny')
# params_files.append('./params/bert/any_cancer_one_split_BERT_linear_tiny')
# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn_sizes_base')
# params_files.append('./params/bert_long/any_cancer_one_split_BERT_cnn_sizes_long')

#bert_tuned
# params_files.append('./params/updated_labels/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_base_frozen')


#JAMA sizes
# params_files.append('./params/updated_labels/JAMA/response_one_split_sizes_JAMA')
# params_files.append('./params/updated_labels/JAMA/anycancer_one_split_sizes_JAMA')

# params_files.append('./params/updated_labels/tfidf/response_one_split_tfidf')

# params_files.append('./params/updated_labels/tfidf/progression_one_split_tfidf')

#clinical BERT

# params_files.append('./params/updated_labels/clinical_bert/response_one_split_CBERT_cnn_sizes_frozen')



#-----------------------progression ----------------------
#BERT original
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_base_frozen')


#BERT Tuned
# params_files.append('./params/updated_labels/tuned_bert_cnn_frozen/progression_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/tuned_bert_cnn_frozen/progression_one_split_BERT_cnn_sizes_base_frozen')

#clinical BERT
# params_files.append('./params/updated_labels/clinical_bert/progression_one_split_CBERT_cnn_sizes_frozen')

#JAMA
# params_files.append('./params/updated_labels/JAMA/progression_one_split_sizes_JAMA')
# params_files.append('./params/updated_labels/tfidf/progression_one_split_tfidf')

#longformer
# params_files.append('./params/updated_labels/longformer/progression_one_split_BERT_cnn_sizes_long_tuned')
# params_files.append('./params/updated_labels/longformer/response_one_split_BERT_cnn_sizes_long_tuned')

# params_files.append('./params/updated_labels/longformer/progression_one_split_BERT_cnn_sizes_long')
# params_files.append('./params/updated_labels/longformer/response_one_split_BERT_cnn_sizes_long')



# params_files.append('./params/updated_labels/tuned_bert_cnn_frozen/any_cancer_one_split_BERT_cnn_sizes_tiny_frozen')
#bert_cnn_arch_size
# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_tiny')
# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_mini')

# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_med')
# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_base')

# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_med')
# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_base')

#bert_cnn_arch_size_frozen
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_mini_frozen')
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_med_frozen')
# params_files.append('./params/updated_labels/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_base_frozen')

#classifier
# params_files.append('./params/updated_labels/bert_classifier/any_cancer_one_split_BERT_sizes_tiny')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_tiny')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_mini_frozen')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_med_frozen')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_base_frozen')



# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_mini')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_med')
# params_files.append('./params/updated_labels/bert_classifier/response_one_split_BERT_sizes_base')

# params_files.append('./params/updated_labels/bert_classifier/progression_one_split_BERT_sizes_tiny_frozen')
# params_files.append('./params/updated_labels/bert_classifier/progression_one_split_BERT_sizes_base_frozen')

params_files.append('./params/updated_labels/bert_classifier/progression_one_split_BERT_sizes_mini_frozen')
params_files.append('./params/updated_labels/bert_classifier/progression_one_split_BERT_sizes_med_frozen')


# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_med_frozen')
# params_files.append('./params/updated_labels/bert_cnn_arch_size/response_one_split_BERT_cnn_sizes_base_frozen')

# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn_sizes_mini_frozen')

# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn_sizes_med_frozen')

# params_files.append('./params/bert/any_cancer_one_split_BERT_cnn_sizes_base_frozen')

# params_files.append('./params/torch_cnn/any_cancer_torch_cnn')
# params_files.append('./params/torch_cnn/any_cancer_torch_cnn2d')
# params_files.append('./params/preprocss/any_cancer_one_split_seq_clean')
# params_files.append('./params/preprocss/any_cancer_one_split_seq_original')
# params_files.append('./params/manual_labels/response_one_split_tfidf')

# params_files.append('./params/manual_labels/any_cancer_crossvalidation_seq')

#response
# params_files.append('./params/manual_labels/response_one_split_tfidf')
# params_files.append('./params/manual_labels/response_one_split_seq')
# params_files.append('./params/manual_labels/response_crossvalidation_seq')
# params_files.append('./params/manual_labels/response_crossvalidation_tfidf')
#
# #progression
# params_files.append('./params/manual_labels/progression_one_split_tfidf')
# params_files.append('./params/manual_labels/progression_one_split_seq')
# params_files.append('./params/manual_labels/progression_crossvalidation_seq')
# params_files.append('./params/manual_labels/progression_crossvalidation_tfidf')

def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for params_file in params_files:
    timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
#     log_dir = params_file.replace('params', 'logs')
    log_dir = join(PROSTATE_LOG_PATH, params_file)
    log_dir = log_dir + timeStamp
    set_logging(log_dir)

    logging.info('random seed %d'%random_seed)
    params_file_full = params_file +'.py'
    params = imp.load_source(params_file, params_file_full)

    if params.pipeline['type'] == 'one_split':
        pipeline = OneSplitPipeline( data_params = params.data,  model_params = params.models, pre_params = params.pre, feature_params = params.features, pipeline_params=params.pipeline, exp_name = log_dir)
    elif params.pipeline['type'] == 'crossvalidation':
        pipeline = CrossvalidationPipeline( data_params = params.data,  model_params = params.models, pre_params = params.pre, feature_params = params.features, pipeline_params=params.pipeline, exp_name = log_dir)

    import timeit

    start = timeit.default_timer()
    pipeline.run()
    stop = timeit.default_timer()
    mins, secs= elapsed_time(start, stop)
    logging.info(f'Epoch Time: {mins}m {secs}s')
