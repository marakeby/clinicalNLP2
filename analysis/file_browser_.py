from os import walk
from os.path import abspath, join
import os
import pandas as pd

base_dir= './gcp_results/updated_labels'

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

    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Jan-27_18-16')
    annotated_files.append(dict(Task='response', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Jan-27_15-36')
    annotated_files.append(dict(Task='response', Model='BERT', Size='tiny', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Jan-27_15-45')
    annotated_files.append(dict(Task='response', Model='BERT', Size='mini', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Jan-27_16-11')
    annotated_files.append(dict(Task='response', Model='BERT', Size='med', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    #progression
    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_06-39')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='tiny', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_06-48')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    '''
    --------- JAMA
    ------------ response_one_split_sizes_JAMA_Feb-24_11-21
    ------------ progression_one_split_sizes_JAMA_Mar-04_03-09
    ------------ anycancer_one_split_sizes_JAMA_Feb-24_11-32
    '''

    sub_dir= join(base_dir, 'JAMA')
    filename= join(sub_dir, 'response_one_split_sizes_JAMA_Feb-24_11-21')
    annotated_files.append(dict(Task='response', Model='JAMA', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

    filename= join(sub_dir, 'progression_one_split_sizes_JAMA_Mar-04_03-09')
    annotated_files.append(dict(Task='progression', Model='JAMA', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

    '''
    --------- longformer
    ------------ response_one_split_BERT_cnn_sizes_long_Mar-06_14-09
    ------------ progression_one_split_BERT_cnn_sizes_long_tuned_Mar-04_05-23
    ------------ response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15
    ------------ progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42
    '''

    sub_dir= join(base_dir, 'longformer')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Mar-06_14-09')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_tuned_Mar-04_05-23')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    '''
    --------- tuned_bert_cnn_frozen
    ------------ progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_13-32
    ------------ progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_13-41
    ------------ response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29
    ------------ response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46
    '''

    sub_dir= join(base_dir, 'tuned_bert_cnn_frozen')
    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46')
    annotated_files.append(dict(Task='response', Model='BERT', Size='tiny', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29')
    annotated_files.append(dict(Task='response', Model='BERT', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))



    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_13-41')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_13-32')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='tiny', Tuned= True, Frozen=True,classifier = 'CNN', file=filename))

    '''
    --------- tfidf
    ------------ response_one_split_tfidf_Feb-24_12-31
    ------------ progression_one_split_tfidf_Mar-04_03-13
    
    '''
    sub_dir= join(base_dir, 'tfidf')
    filename= join(sub_dir, 'response_one_split_tfidf_Feb-24_12-31')
    annotated_files.append(dict(Task='response', Model='tfidf', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'progression_one_split_tfidf_Mar-08_07-53')
    annotated_files.append(dict(Task='progression', Model='tfidf', Size='NA', Tuned= 'NA', Frozen="NA",classifier = 'CNN', file=filename))


    '''
    --------- clinical_bert
    ------------ progression_one_split_CBERT_cnn_sizes_frozen_Mar-03_20-25
    ------------ response_one_split_CBERT_cnn_sizes_frozen_Feb-24_13-02
    '''

    sub_dir= join(base_dir, 'clinical_bert')
    filename= join(sub_dir, 'response_one_split_CBERT_cnn_sizes_frozen_Feb-24_13-02')
    annotated_files.append(dict(Task='response', Model='clinical BERT', Size='base', Tuned= 'NA', Frozen=True, classifier = 'CNN', file=filename))


    filename= join(sub_dir, 'progression_one_split_CBERT_cnn_sizes_frozen_Mar-03_20-25')
    annotated_files.append(dict(Task='progression', Model='clinical BERT', Size='base', Tuned= 'NA', Frozen=True, classifier = 'CNN', file=filename))

    return  pd.DataFrame(annotated_files)

if __name__ == "__main__":
    print(get_files())

