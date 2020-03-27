from rnn.timeseries import Preprocessing

path = 'dataset/test.csv'
seq_length = 14
predict = 7
split = 0.7
a = Preprocessing()
a.setdata(path, seq_length, predict, split)
train_data, test_data, train_label, test_label = a.preprocessing()


print('train_data의 크기: ', train_data.shape)
print('test_data의 크기: ', test_data.shape)
print('train_label의 크기: ', train_label.shape)
print('test_label의 크기: ', test_label.shape)


train_data[0]


import pandas as pd
import numpy as np
dataset = pd.read_csv(path, index_col=None, header=None)


data = dataset.values.astype(np.float32)
label = dataset.iloc[:, -1:].values.astype(np.float32)

print(data[1:15])
label[21]