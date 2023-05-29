
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


# arch size
# params_files.append('./params/rerun/unfrozen/progression_BERT_base_linear_tuned_small')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_mini_unfrozen_linear')

# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_tiny_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_med_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_base_unfrozen_linear')

# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_tiny_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_mini_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_med_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_base_unfrozen_linear')


# params_files.append('./params/revision2/new_splits/longformer/unfrozen/response_longformer_base_unfrozen_linear')
# params_files.append('./params/revision2/new_splits/longformer/unfrozen/progression_longformer_base_unfrozen_linear')

# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_tiny_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_mini_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_med_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/response_BERT_base_unfrozen_linear_try1')

# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_tiny_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_mini_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_med_unfrozen_linear_try1')
# params_files.append('./params/revision2/new_splits/arch_size/unfrozen/progression_BERT_base_unfrozen_linear_try1')


#classifier
params_files.append('./params/revision2/new_splits/classifier/unfrozen/response_DFCI_BERT_sizes_base_unfrozen_linear_try1')
params_files.append('./params/revision2/new_splits/classifier/unfrozen/progression_DFCI_BERT_sizes_base_unfrozen_linear_try1')

#tfidf
# params_files.append('./params/revision2/new_splits/tfidf/progression_one_split_tfidf')
# params_files.append('./params/revision2/new_splits/tfidf/response_one_split_tfidf')



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
