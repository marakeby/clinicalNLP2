from data.manual_labels.data_reader import ManualLabels
import pandas as pd
data = ManualLabels(outcome = 'any_cancer', text='NARR+IMPRESS')
# data = ManualLabels(outcome = 'any_cancer', text='NARR')
# data = ManualLabels(outcome = 'any_cancer', text='IMPRESS')
x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()

# print type(x_train)
# print (x_train.shape)
# print (x_train.head)
# print (x_validate.shape)
# print (x_test.shape)
# print (type(x_train))
# print info_train.head()
# print (y_train.head())
# print (info_train.head())
#
def format_data_set(x, y, info):
    df = info.copy()
    df['user_id'] = info.index
    df['label'] = y
    df['alpha'] = ['a'] * df.shape[0]
    df['text'] = x.replace('\n',' ')
    df = df[['user_id', 'label', 'alpha', 'text']]
    return df

df = format_data_set(x_train, y_train, info_train)
null_data = df[df.isnull().any(axis=1)]
print( null_data)
# print (df.shape)
df.to_csv('bert_format/any_cancer/train.tsv', sep='\t', index=False, header=False)
df.to_json('bert_format/any_cancer/train.json')
# df.to_csv('bert_format/any_cancer/train.csv', sep=',', index=False, header=False)
#
df = format_data_set(x_test, y_test, info_test)
# print (df.shape)
null_data = df[df.isnull().any(axis=1)]
print (null_data)
df.to_csv('bert_format/any_cancer/test.tsv', sep='\t', index=False, header=False)
df.to_json('bert_format/any_cancer/test.json')
# df.to_csv('bert_format/any_cancer/test.csv', sep=',', index=False, header=False)
#
df = format_data_set(x_validate, y_validate, info_validate)
# print (df.shape)
null_data = df[df.isnull().any(axis=1)]
print (null_data)
df.to_csv('bert_format/any_cancer/dev.tsv', sep='\t', index=False, header=False)
df.to_json('bert_format/any_cancer/dev.json')
dd = pd.read_csv('bert_format/any_cancer/dev.tsv', sep='\t',  header=None)
print (dd.head())
# df.to_csv('bert_format/any_cancer/dev.csv', sep=',', index=False, header=False)
#
