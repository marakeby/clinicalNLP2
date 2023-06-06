from os import walk
from os.path import abspath, join
import os
import pandas as pd
from os.path import join
from config_path import BASE_PATH

# base_dir= './gcp_results/updated_labels'
# base_dir= './updated_labels'
base_dir = join(BASE_PATH, '_gcp_revision')
# base_dir= '.'

'''
Task {progression| response} -> classification task
Model: {BERT| JAMA| TFIDF| clinical BERT| longfromer } -> BERT model arch 
Size: {BASE| TINY| MED|MINI, NA} -> BERT model size
Tuned: {TUNED | Original} -> Tuned: DFCIIMage model (model tuned on DFCI data), Original: not tuned 
Forzen: {forzen| non-forzen|NA} ->  BERT model is either frozen or not frozen
Classifier : {CNN|RNN|Linear|NA} -> classification head on top of BERT model
'''
annotated_files=[]

def get_clinical():
    sub_dir = join(base_dir, 'clinical/unfrozen')
    # filename= join(sub_dir, 'progression_clinical_BERT_linear_Mar-07_10-16')
    filename = join(sub_dir, 'progression_clinical_BERT_base_unfrozen_linear_earlystop_May-24_20-10')
    annotated_files.append(
        dict(Task='progression', Model='clinical BERT', Size='base', Tuned=False, Frozen=False, classifier='Linear',
             file=filename))

    sub_dir = join(base_dir, 'clinical/unfrozen')
    filename = join(sub_dir, 'response_clinical_BERT_base_unfrozen_linear_earlystop_May-24_18-06')
    annotated_files.append(
        dict(Task='response', Model='clinical BERT', Size='base', Tuned=False, Frozen=False, classifier='Linear',
             file=filename))

    sub_dir = join(base_dir, 'clinical/frozen')
    # filename = join(sub_dir, 'progression_clinical_BERT_base_frozen_cnn_earlystop_May-24_14-47')
    filename = join(sub_dir, 'progression_clinical_BERT_base_frozen_cnn_Jun-06_08-59')

    annotated_files.append(
        dict(Task='progression', Model='clinical BERT', Size='base', Tuned=False, Frozen=True, classifier='CNN',
             file=filename))

    sub_dir = join(base_dir, 'clinical/frozen')
    filename = join(sub_dir, 'response_clinical_BERT_base_frozen_cnn_Jun-06_07-57')

    # filename = join(sub_dir, 'response_clinical_BERT_base_frozen_cnn_earlystop_May-24_13-49')
    annotated_files.append(
        dict(Task='response', Model='clinical BERT', Size='base', Tuned=False, Frozen=True, classifier='CNN',
             file=filename))

def get_DFCI_BERT():
    sub_dir = join(base_dir, 'DFCI_BERT/unfrozen')
    filename = join(sub_dir, 'progression_DFCI_BERT_base_unfrozen_linear_earlystop_May-24_05-51')
    annotated_files.append(
        dict(Task='progression', Model='BERT', Size='base', Tuned=True, Frozen=False, classifier='Linear',
             file=filename))

    sub_dir = join(base_dir, 'DFCI_BERT/unfrozen')
    filename = join(sub_dir, 'response_DFCI_BERT_base_unfrozen_linear_earlystop_May-24_03-47')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='base', Tuned=True, Frozen=False, classifier='Linear',
             file=filename))

    sub_dir = join(base_dir, 'DFCI_BERT/frozen')
    # filename = join(sub_dir, 'progression_DFCI_BERT_base_frozen_cnn_earlystop_May-24_10-18')
    filename = join(sub_dir, 'progression_DFCI_BERT_base_frozen_cnn_May-30_12-07')
    annotated_files.append(
        dict(Task='progression', Model='BERT', Size='base', Tuned=True, Frozen=True, classifier='CNN',
             file=filename))

    sub_dir = join(base_dir, 'DFCI_BERT/frozen')
    # filename = join(sub_dir, 'response_DFCI_BERT_base_frozen_cnn_earlystop_May-24_09-12')
    filename = join(sub_dir, 'response_DFCI_BERT_base_frozen_cnn_May-30_13-09')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='base', Tuned=True, Frozen=True, classifier='CNN',
             file=filename))


def get_JAMA():
    sub_dir = join(base_dir, 'JAMA')
    # filename = join(sub_dir, 'response_one_split_sizes_JAMA_May-24_01-42')
    filename = join(sub_dir, 'response_one_split_sizes_JAMA_Jun-05_19-34')
    annotated_files.append(dict(Task='response', Model='CNN', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

    sub_dir = join(base_dir, 'JAMA')
    # filename = join(sub_dir, 'progression_one_split_sizes_JAMA_May-24_01-37')
    filename = join(sub_dir, 'progression_one_split_sizes_JAMA_Jun-05_19-29')
    annotated_files.append(dict(Task='progression', Model='CNN', Size='NA', Tuned= 'NA', Frozen='NA',classifier = 'CNN', file=filename))

def get_arch_size():


    def get_progression_frozen():
        sub_dir = join(base_dir, 'arch_size/frozen')
        # filename= join(sub_dir, 'progression_BERT_base_frozen_cnn_earlystop_May-26_04-36')
        filename= join(sub_dir, 'progression_BERT_base_frozen_cnn_May-26_21-54')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='base', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename= join(sub_dir, 'progression_BERT_med_frozen_cnn_earlystop_May-26_04-32')
        filename= join(sub_dir, 'progression_BERT_med_frozen_cnn_May-26_21-47')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename= join(sub_dir, 'progression_BERT_mini_frozen_cnn_earlystop_May-26_04-30')
        filename= join(sub_dir, 'progression_BERT_mini_frozen_cnn_May-26_21-45')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='mini', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename= join(sub_dir, 'progression_BERT_tiny_frozen_cnn_earlystop_May-26_04-29')
        filename= join(sub_dir, 'progression_BERT_tiny_frozen_cnn_May-26_21-39')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

    def get_response_frozen():
        sub_dir = join(base_dir, 'arch_size/frozen')
        # filename = join(sub_dir, 'response_BERT_base_frozen_cnn_earlystop_May-26_04-14')
        filename = join(sub_dir, 'response_BERT_base_frozen_cnn_May-26_22-20')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='base', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_med_frozen_cnn_earlystop_May-26_04-08')
        filename = join(sub_dir, 'response_BERT_med_frozen_cnn_May-26_22-13')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_mini_frozen_cnn_earlystop_May-26_04-06')
        filename = join(sub_dir, 'response_BERT_mini_frozen_cnn_May-26_22-11')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='mini', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_tiny_frozen_cnn_earlystop_May-26_04-05')
        filename = join(sub_dir, 'response_BERT_tiny_frozen_cnn_May-26_21-34')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN',
                 file=filename))

    def get_progression_unfrozen():
        sub_dir = join(base_dir, 'arch_size/unfrozen')
        filename = join(sub_dir, 'progression_BERT_base_unfrozen_linear_earlystop_May-26_05-33')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='base', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        filename = join(sub_dir, 'progression_BERT_med_unfrozen_linear_earlystop_May-26_05-25')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='med', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        filename = join(sub_dir, 'progression_BERT_mini_unfrozen_linear_earlystop_May-26_05-23')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='mini', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        filename = join(sub_dir, 'progression_BERT_tiny_unfrozen_linear_earlystop_May-26_05-21')
        annotated_files.append(
            dict(Task='progression', Model='BERT', Size='tiny', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

    def get_response_unfrozen():
        sub_dir = join(base_dir, 'arch_size/unfrozen')
        # filename = join(sub_dir, 'response_BERT_base_unfrozen_linear_earlystop_May-26_05-01')
        filename = join(sub_dir, 'response_BERT_base_unfrozen_linear_try1_May-27_13-23')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='base', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_med_unfrozen_linear_earlystop_May-26_04-52')
        filename = join(sub_dir, 'response_BERT_med_unfrozen_linear_try1_May-27_13-15')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='med', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_mini_unfrozen_linear_earlystop_May-26_04-49')
        filename = join(sub_dir, 'response_BERT_mini_unfrozen_linear_try1_May-27_13-11')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='mini', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

        # filename = join(sub_dir, 'response_BERT_tiny_unfrozen_linear_earlystop_May-26_04-48')
        filename = join(sub_dir, 'response_BERT_tiny_unfrozen_linear_try1_May-27_13-09')
        annotated_files.append(
            dict(Task='response', Model='BERT', Size='tiny', Tuned=False, Frozen=False, classifier='Linear',
                 file=filename))

    get_response_frozen()
    get_response_unfrozen()
    get_progression_frozen()
    get_progression_unfrozen()

def get_longformer():
    '''
    --------- longformer
    ------------ response_one_split_BERT_cnn_sizes_long_Mar-06_14-09
    ------------ progression_one_split_BERT_cnn_sizes_long_tuned_Mar-04_05-23
    ------------ response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15
    ------------ progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42
    '''

    sub_dir = join(base_dir, 'longformer/unfrozen')

    # sub_dir = join(base_dir, 'unfrozen/longformer')
    # filename= join(sub_dir, 'progression_longformer_unfrozen_linear_earlystop_May-26_10-30')
    filename= join(sub_dir, 'progression_longformer_base_unfrozen_linear_May-30_16-13')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= False, Frozen=False, classifier = 'Linear', file=filename))


    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Mar-06_14-09')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_Aug-28_12-24')
    # filename= join(sub_dir, 'response_longformer_unfrozen_linear_earlystop_May-26_10-17')
    filename= join(sub_dir, 'response_longformer_base_unfrozen_linear_May-30_14-11')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= False, Frozen=False,classifier = 'Linear', file=filename))

    sub_dir = join(base_dir, 'longformer/frozen')

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Mar-05_23-42')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_long_Aug-29_18-56')
    filename= join(sub_dir, 'progression_longformer_frozen_cnn_May-26_20-19')
    annotated_files.append(dict(Task='progression', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Mar-05_09-15')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_long_tuned_Aug-29_03-41')
    filename= join(sub_dir, 'response_longformer_frozen_cnn_May-26_18-36')
    annotated_files.append(dict(Task='response', Model='longformer', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))




def get_tfidf():
    '''
        --------- tfidf
        ------------ response_one_split_tfidf_Feb-24_12-31
        ------------ progression_one_split_tfidf_Mar-04_03-13

        '''
    sub_dir = join(base_dir, 'tfidf')
    # filename= join(sub_dir, 'response_one_split_tfidf_Feb-24_12-31')
    # filename= join(sub_dir, 'response_one_split_tfidf_Aug-24_17-43')
    # filename= join(sub_dir, 'response_one_split_tfidf_Nov-16_14-24')
    filename= join(sub_dir, 'response_one_split_tfidf_May-28_15-03')
    annotated_files.append(
        dict(Task='response', Model='TF-IDF', Size='NA', Tuned='NA', Frozen='NA', classifier='NA', file=filename))

    # filename= join(sub_dir, 'progression_one_split_tfidf_Mar-08_07-53')
    # filename= join(sub_dir, 'progression_one_split_tfidf_Aug-24_17-44')
    filename= join(sub_dir, 'progression_one_split_tfidf_May-28_15-00')
    annotated_files.append(
        dict(Task='progression', Model='TF-IDF', Size='NA', Tuned='NA', Frozen="NA", classifier='NA', file=filename))


def get_classifier():
    sub_dir = join(base_dir, 'classifier/frozen')
    filename = join(sub_dir, 'progression_DFCI_BERT_sizes_base_frozen_May-25_09-59')
    annotated_files.append( dict(Task='progression', Model='BERT', Size='base', Tuned=True, Frozen=False, classifier='NA',
             file=filename))

    sub_dir = join(base_dir, 'classifier/frozen')
    filename = join(sub_dir, 'response_DFCI_BERT_sizes_base_frozen_May-25_13-39')
    annotated_files.append( dict(Task='response', Model='BERT', Size='base', Tuned=True, Frozen=False, classifier='NA',
             file=filename))

    return

def get_zero_shot():
    sub_dir = join(base_dir, 'zero_shot')
    filename= join(sub_dir, 'google__flan-t5-xxl-response_0MRNs_0kTrainSteps')
    annotated_files.append(dict(Task='response', Model='FlanT5-zeroshot', Size='xxl', Tuned= 'NA', Frozen='NA', classifier = 'NA', file=filename))

    filename= join(sub_dir, 'google__flan-t5-xxl-progression_0MRNs_0kTrainSteps')
    annotated_files.append(dict(Task='progression', Model='FlanT5-zeroshot', Size='xxl', Tuned= 'NA', Frozen='NA', classifier = 'NA', file=filename))
    return

def get_models():
    get_clinical()
    get_DFCI_BERT()
    get_JAMA()
    # get_classifier()
    get_arch_size() #replace
    get_longformer() #replace
    get_tfidf()
    get_zero_shot()

    return  pd.DataFrame(annotated_files)

'''

    sub_dir = join(base_dir, 'unfrozen/tune')
    # filename= join(sub_dir, 'progression_BERT_base_tuned_linear_Mar-06_19-28')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= True, Frozen=False, classifier = 'Linear', file=filename))


    sub_dir= join(base_dir, 'bert_cnn_arch_size_frozen')
    # sub_dir= join(base_dir, 'bert_cnn_arch_size_frozen_truncation')

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Jan-27_18-16')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Aug-25_15-29')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_base_frozen_Oct-06_03-55')
    annotated_files.append(dict(Task='response', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Jan-27_16-11')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Aug-25_22-42')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_med_frozen_Oct-06_10-46')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Jan-27_15-45')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Aug-26_00-55')
    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_mini_frozen_Oct-06_12-54')
    annotated_files.append(dict(Task='response', Model='BERT', Size='mini', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))


    # filename= join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Jan-27_15-36')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Aug-26_01-24')
    # filename = join(sub_dir, 'response_one_split_BERT_cnn_sizes_tiny_frozen_Oct-06_13-21')
    annotated_files.append(
        dict(Task='response', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    #progression

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_06-48')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Aug-31_14-25')
    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_base_frozen_Oct-05_18-28')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='base', Tuned= False, Frozen=True,classifier = 'CNN', file=filename))

    #--- new
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Jun-02_15-54')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Aug-25_12-47')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_med_frozen_Oct-06_01-20')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='med', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Jun-02_15-26')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Aug-25_15-01')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_mini_frozen_Oct-06_03-28')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='mini', Tuned=False, Frozen=True, classifier='CNN', file=filename))

    # filename= join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_06-39')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Aug-25_12-10')
    # filename = join(sub_dir, 'progression_one_split_BERT_cnn_sizes_tiny_frozen_Oct-05_15-12')
    annotated_files.append(dict(Task='progression', Model='BERT', Size='tiny', Tuned=False, Frozen=True, classifier='CNN', file=filename))




    
    --------- tuned_bert_cnn_frozen
    ------------ progression_one_split_BERT_cnn_sizes_tiny_frozen_Mar-03_13-32
    ------------ progression_one_split_BERT_cnn_sizes_base_frozen_Mar-03_13-41
    ------------ response_one_split_BERT_cnn_sizes_base_frozen_Feb-24_03-29
    ------------ response_one_split_BERT_cnn_sizes_tiny_frozen_Feb-14_15-46
    


'''




if __name__ == "__main__":
    # print(get_models())
    print(get_models().columns)

