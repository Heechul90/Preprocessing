from timeseries.prediction_preprocessing import Preprocessing


path = 'dataset/prediction/test1.csv'
seq_length = 14
predict = 7
split = 0.7
a = Preprocessing()
a.setdata(path, seq_length, predict, split)
train_data, test_data, train_label, test_label = a.prediction()


print('train_data의 크기: ', train_data.shape)
print('test_data의 크기: ', test_data.shape)
print('train_label의 크기: ', train_label.shape)
print('test_label의 크기: ', test_label.shape)


########################################################################################################################
from timeseries.autoencoder_preprocessing import Preprocessing

path = 'dataset/autoencoder/train_test.csv'
validation = 0.6

a = Preprocessing()
a.setdata(path, validation)
train_data, validation_data = a.autoencoder()
train_data