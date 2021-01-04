import pandas as pd
import os
from os.path import  join

from config_path import DATA_PATH

dirr= join(DATA_PATH, 'updated_labels')
filename= join(dirr,'geekfest_files_labeled_imaging_reports_manual_dedup_10-8-19.feather' )
df = pd.read_feather(filename)
print (df.shape)

