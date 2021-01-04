import logging
import datetime
import os
import sys

def set_logging(filename):
    # base_folder ='../logs'
    # timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

    # filename, file_extension = os.path.splitext(filename)
    # filename = os.path.join(base_folder,filename+ timeStamp+'.log')
    if not os.path.exists(filename):
        os.makedirs(filename)

    filename = os.path.join(filename,  'log.log')
    logging.basicConfig(filename = filename,
                        filemode='w',
                        format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                        datefmt='%m/%d %I:%M',
                        level=logging.INFO) # or logging.DEBUG
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('setting logs')