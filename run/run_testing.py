
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

from config_path import CNLP_LOG_PATH


#JAMA sizes
# params_files.append('./params/cnlp_testing/JAMA/response_one_split_sizes_JAMA')
# params_files.append('./params/cnlp_testing/JAMA/progression_one_split_sizes_JAMA')

## TF-IDF
# params_files.append('./params/cnlp_testing/tfidf/response_one_split_tfidf')
# params_files.append('./params/cnlp_testing/tfidf/progression_one_split_tfidf')

# params_files.append('./params/cnlp_testing/tfidf/response_one_split_tfidf_nn')
# params_files.append('./params/cnlp_testing/tfidf/progression_one_split_tfidf_nn')
# params_files.append('./params/cnlp_testing/tfidf/progression_one_split_tfidf_compare')

#clinical BERT
# params_files.append('./params/cnlp_testing/clinical_bert/response_one_split_CBERT_cnn_sizes_frozen')
# params_files.append('./params/cnlp_testing/clinical_bert/progression_one_split_CBERT_cnn_sizes_frozen')

#sizes
#    'bert_cnn_arch_size_frozen'
# 
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_tiny_frozen')

# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_med_frozen')
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/progression_one_split_BERT_cnn_sizes_mini_frozen')


# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_med_frozen')
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_mini_frozen')
# params_files.append('./params/cnlp_testing/bert_cnn_arch_size_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen')

#
#   --------- tuned_bert_cnn_frozen
#
    
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen/progression_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen/progression_one_split_BERT_cnn_sizes_tiny_frozen_truncated')

# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen/progression_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen/response_one_split_BERT_cnn_sizes_tiny_frozen')

# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen_truncated/progression_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen_truncated/response_one_split_BERT_cnn_sizes_base_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen_truncated/progression_one_split_BERT_cnn_sizes_tiny_frozen')
# params_files.append('./params/cnlp_testing/tuned_bert_cnn_frozen_truncated/response_one_split_BERT_cnn_sizes_tiny_frozen')

##. longformer

# params_files.append('./params/cnlp_testing/longformer/response_one_split_BERT_cnn_sizes_long')
# params_files.append('./params/cnlp_testing/longformer/response_one_split_BERT_cnn_sizes_long_tuned')
# params_files.append('./params/cnlp_testing/longformer/progression_one_split_BERT_cnn_sizes_long')
# params_files.append('./params/cnlp_testing/longformer/progression_one_split_BERT_cnn_sizes_long_tuned')

##. longformer

# params_files.append('./params/cnlp_testing/longformer_truncated/response_one_split_BERT_cnn_sizes_long_tuned')
# params_files.append('./params/cnlp_testing/longformer_truncated/response_one_split_BERT_cnn_sizes_long')

# params_files.append('./params/cnlp_testing/longformer_truncated/progression_one_split_BERT_cnn_sizes_long')
# params_files.append('./params/cnlp_testing/longformer_truncated/progression_one_split_BERT_cnn_sizes_long_tuned')


##. Classifier

# params_files.append('./params/cnlp_testing/bert_classifier/progression_one_split_BERT_sizes_base_frozen_tuned')
# params_files.append('./params/cnlp_testing/bert_classifier/response_one_split_BERT_sizes_base_frozen_tuned')
# params_files.append('./params/cnlp_testing/bert_classifier/progression_one_split_BERT_sizes_tiny_frozen_tuned')


##. unfrozen
# params_files.append('./params/cnlp_testing/unfrozen/progression_BERT_tiny_frozen_unfrozen_linear')
params_files.append('./params/cnlp_testing/unfrozen/progression_BERT_base_frozen_unfrozen_linear')


def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for params_file in params_files:
    timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
#     log_dir = params_file.replace('params', 'logs')
    log_dir = join(CNLP_LOG_PATH, params_file)
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
    logging.info(f'Total Time: {mins}m {secs}s')
