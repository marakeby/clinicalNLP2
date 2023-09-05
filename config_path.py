from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, 'data')

RUN_PATH = join(BASE_PATH, 'run')

LOG_PATH = join(BASE_PATH, 'logs')

# CNLP_LOG_PATH = join(LOG_PATH, 'cnlp_testing')
CNLP_LOG_PATH = join(LOG_PATH, '_gcp_revision')
PARAMS_PATH = join(RUN_PATH, 'params')


# TEST_RESULTS_PATH  = join(BASE_PATH, '_cnlp_results')
# TEST_RESULTS_PATH  = join(BASE_PATH, '_gcp_logs_unfrozen/cnlp_testing/params/cnlp_testing')
TEST_RESULTS_PATH  = join(BASE_PATH, '_gcp_revision')
# PLOTS_PATH  = join(BASE_PATH, '_cnlp_plots')
# PLOTS_PATH  = join(BASE_PATH, '_cnlp_plots_unfrozen')
PLOTS_PATH  = join(BASE_PATH, '_plots_revision')