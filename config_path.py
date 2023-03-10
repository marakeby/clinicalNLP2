from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, 'data')

RUN_PATH = join(BASE_PATH, 'run')

LOG_PATH = join(BASE_PATH, 'logs')

CNLP_LOG_PATH = join(LOG_PATH, 'cnlp_testing')
PARAMS_PATH = join(RUN_PATH, 'params')


TEST_RESULTS_PATH  = join(BASE_PATH, '_cnlp_results')
PLOTS_PATH  = join(BASE_PATH, '_cnlp_plots')