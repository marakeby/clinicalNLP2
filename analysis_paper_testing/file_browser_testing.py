from os import walk
from os.path import abspath, join
import os
import pandas as pd

# base_dir= './gcp_results/updated_labels'
base_dir= './updated_labels'

'''
Task {progression| response}
Model: {BERT| JAMA| TFIDF| clinical BERT| longfromer }
Size: {BASE| TINY| MED|MINI, NA}
Tuned: {TUNED | Original}
Forzen: {forzen| non-forzen|NA}
Classifier : {CNN|RNN|Linear|NA}
'''

def get_models():
    #sizes
    'bert_cnn_arch_size_frozen'

    annotated_files=[]
    sub_dir= join(base_dir, 'bert_cnn_arch_size_frozen')
    # sub_dir= join(base_dir, 'bert_cnn_arch_size_frozen_truncation')

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Jan-27_18-16')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Aug-25_15-29')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Oct-06_03-55')
    annotated_files.append(dict(Task='response', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Jan-27_16-11')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Aug-25_22-42')
    filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Oct-06_10-46')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Jan-27_15-45')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Aug-26_00-55')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Oct-06_12-54')
    annotated_files.append(dict(Task='response', Model='BERT', Size='mini', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Jan-27_15-36')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Aug-26_01-24')
    filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Oct-06_13-21')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    #progression

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_06-48')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Aug-31_14-25')
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Oct-05_18-28')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    #--- new
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Jun-02_15-54')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Aug-25_12-47')
    filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Oct-06_01-20')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Jun-02_15-26')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Aug-25_15-01')
    filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Oct-06_03-28')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='mini', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_06-39')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Aug-25_12-10')
    filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Oct-05_15-12')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    '''
    --------- JAMA
    ------------ response_one_split_sizes_JAMA_Feb-24_11-21
    ------------ progression_one_split_sizes_JAMA_Mar-04_03-09
    ------------ anycancer_one_split_sizes_JAMA_Feb-24_11-32
    '''

    sub_dir= join(base_dir, 'JAMA')
    # filename= join(sub_dir, 'response_one_split_sizes_JAMA_Feb-24_11-21')
    # filename= join(sub_dir, 'response_one_split_sizes_JAMA_Aug-24_17-21')
    filename= join(sub_dir, 'response_one_split_sizes_JAMA_Nov-16_14-28')
    annotated_files.append(dict(Task='response', Model='CNN', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

    # filename= join(sub_dir, 'progression_one_split_sizes_JAMA_Mar-04_03-09')
    # filename= join(sub_dir, 'progression_one_split_sizes_JAMA_Aug-24_17-25')
    filename= join(sub_dir, 'progression_one_split_sizes_JAMA_Nov-16_14-32')
    annotated_files.append(dict(Task='progression', Model='CNN', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

    '''
    --------- longformer
    ------------ response_one_split_BERT_cnn_sizes_long_Mar-06_14-09
    ------------ progression_one_split_BERT_cnn_sizes_long_tuned_Mar-04_05-23
    ------------ response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15
    ------------ progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42
    '''

    # sub_dir= join(base_dir, 'longformer')
    sub_dir= join(base_dir, 'longformer_truncated')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Mar-06_14-09')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Aug-28_12-24')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Oct-20_08-34')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Aug-29_18-56')
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Oct-20_23-07')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Aug-29_03-41')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Oct-19_18-00')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_tuned_Mar-04_05-23')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_tuned_Aug-30_10-10')
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_tuned_Oct-21_13-41')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    '''
    --------- tuned_bert_cnn_frozen
    ------------ progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_13-32
    ------------ progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_13-41
    ------------ response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29
    ------------ response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46
    '''

    # sub_dir= join(base_dir, 'tuned_bert_cnn_frozen')
    sub_dir= join(base_dir, 'tuned_bert_cnn_frozen_truncated')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Aug-27_09-26')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Oct-23_11-40')
    annotated_files.append(dict(Task='response', Model='BERT', Size='tiny', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Aug-27_02-13')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Oct-23_04-40')
    annotated_files.append(dict(Task='response', Model='BERT', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_13-41')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Aug-26_18-59')
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Oct-22_21-48')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_13-32')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Aug-26_18-51')
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Oct-23_11-32')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='tiny', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    '''
    --------- tfidf
    ------------ response_one_split_tfidf_Feb-24_12-31
    ------------ progression_one_split_tfidf_Mar-04_03-13
    
    '''
    sub_dir= join(base_dir, 'tfidf')
    # filename= join(sub_dir, 'response_one_split_tfidf_Feb-24_12-31')
    # filename= join(sub_dir, 'response_one_split_tfidf_Aug-24_17-43')
    filename= join(sub_dir, 'response_one_split_tfidf_Nov-16_14-24')
    annotated_files.append(dict(Task='response', Model='TF-IDF', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'progression_one_split_tfidf_Mar-08_07-53')
    # filename= join(sub_dir, 'progression_one_split_tfidf_Aug-24_17-44')
    filename= join(sub_dir, 'progression_one_split_tfidf_Nov-16_14-25')
    annotated_files.append(dict(Task='progression', Model='TF-IDF', Size='NA', Tuned= 'NA', Frozen="NA",classifier = 'CNN', file=filename))


    '''
    --------- clinical_bert
    ------------ progression_one_split_CBERT_cnn_sizes_frozen_Mar-03_20-25
    ------------ response_one_split_CBERT_cnn_sizes_frozen_Feb-24_13-02
    '''

    sub_dir= join(base_dir, 'clinical_bert')
    # filename= join(sub_dir, 'response_one_split_CBERT_cnn_sizes_frozen_Feb-24_13-02')
    # filename= join(sub_dir, 'response_one_split_CBERT_cnn_sizes_frozen_Aug-24_18-07')
    filename= join(sub_dir, 'response_one_split_CBERT_cnn_sizes_frozen_Oct-25_15-26')
    annotated_files.append(dict(Task='response', Model='clinical BERT', Size='base', Tuned= 'NA', Frozen=True, classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'progression_one_split_CBERT_cnn_sizes_frozen_Mar-03_20-25')
    # filename= join(sub_dir, 'progression_one_split_CBERT_cnn_sizes_frozen_Aug-25_01-21')
    filename= join(sub_dir, 'progression_one_split_CBERT_cnn_sizes_frozen_Oct-25_22-18')
    annotated_files.append(dict(Task='progression', Model='clinical BERT', Size='base', Tuned= 'NA', Frozen=True, classifier = 'CNN', file=filename))

    return  pd.DataFrame(annotated_files)

if __name__ == "__main__":
    # print(get_models())
    print(get_models().columns)

