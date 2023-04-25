from data.updated_labels.data_reader import UpdatedLabels

# data = ManualLabels(outcome = 'any_cancer', text='IMPRESS')
data =[{'id':'any_cancer_IMPRESS','type': 'updated_label' , 'params': {'outcome': 'response', 'text': 'NARR+IMPRESS', 'training_split':0 , 'cloud':True}}]

# data = ManualLabels(outcome = 'any_cancer')
data = UpdatedLabels(**data[0]['params'])
x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()

print ('x_train', x_train.shape)
print (type(x_train))
print ('x_train', x_validate.shape)
print ('x_train', x_test.shape)



total = y_train.shape[0]
pos = y_train.sum()
percent = 100.*pos/total
print ('y_train total {}, pos {} , precent {} %'.format( total, pos, percent))

total = y_test.shape[0]
pos = y_test.sum()
percent = 100.*pos/total
print ('y_test total {}, pos {} , precent {} %'.format(total, pos, percent))

total = y_validate.shape[0]
pos = y_validate.sum()
percent = 100.*pos/total
print ('y_validate total {}, pos {} , precent {} %'.format(total, pos, percent))

