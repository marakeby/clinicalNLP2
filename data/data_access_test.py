from data.data_access import Data
import logging
from matplotlib import pyplot as plt

# params= {'type': 'claims', 'params': {'data_type': 'seq' }}
# params= {'type': 'claims', 'params': {'data_type': 'seq' }}
# params= {'type': 'claims', 'params': {'data_type': '3gram_freq' }}
params= {'type': 'manual_label', 'params': {'outcome': 'any_cancer' }}
data = Data(**params)

x, y, info, columns = data.get_data()
print columns
print len(x), y.shape, x.shape, info.shape, len(columns)
print x[0]

# x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()


# print [x for x in y.columns if x.startswith('BRAF')]
# plt.hist(y['BRAF'])
# plt.savefig('BRAF.png')
# print x[0,:]

