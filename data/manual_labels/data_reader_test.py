from data.manual_labels.data_reader import ManualLabels

# data = ManualLabels(outcome = 'any_cancer', text='IMPRESS')
data =[{'id':'any_cancer_IMPRESS','type': 'manual_label' , 'params': {'outcome': 'any_cancer', 'text': 'NARR+IMPRESS', 'training_split':0}}]

# data = ManualLabels(outcome = 'any_cancer')
data = ManualLabels(**data[0]['params'])
x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()

print ('x_train', x_train.shape)
print (type(x_train))
print ('x_train', x_validate.shape)
print ('x_train', x_test.shape)

# print 'info', info_train.head()
# print 'info', info_train.shape
# print 'info_test', info_test.shape
# print 'y_train', sum(y_train)/(len(y_train) +0.)
# print 'y_validate', sum(y_validate)/(len(y_validate) +0.)
# print 'y_test', sum(y_test)/(len(y_test) +0.)



# data = ManualLabels(outcome = 'any_cancer')
# x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()
#
# print 'info', info_train.shape, info_validate.shape, info_test.shape
# print 'info', x_train.shape, x_validate.shape, x_test.shape
# print 'info', y_train.shape, y_validate.shape, y_test.shape
# # print 'info', info_train.head()
# print 'info_test', info_test.shape
# print 'any_cancer ---------------'
# print 'y_train', y_train.shape, sum(y_train)/(len(y_train) +0.)
# print 'y_validate',y_validate.shape,  sum(y_validate)/(len(y_validate) +0.)
# print 'y_test', y_test.shape, sum(y_test)/(len(y_test) +0.)
#
#
# print 'response ---------------'
# data = ManualLabels(outcome = 'response')
# x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()
# print 'info', info_train.shape
# print 'info_test', info_test.shape
# print 'y_train', y_train.shape, sum(y_train)/(len(y_train) +0.)
# print 'y_validate',y_validate.shape,  sum(y_validate)/(len(y_validate) +0.)
# print 'y_test', y_test.shape, sum(y_test)/(len(y_test) +0.)
#
# print 'progression ---------------'
# data = ManualLabels(outcome = 'progression')
# x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()
# print 'info', info_train.shape
# print 'info_test', info_test.shape
# print 'y_train', y_train.shape, sum(y_train)/(len(y_train) +0.)
# print 'y_validate',y_validate.shape,  sum(y_validate)/(len(y_validate) +0.)
# print 'y_test', y_test.shape, sum(y_test)/(len(y_test) +0.)