
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
import utils



########################################################################################################################
from timeseries.autoencoder_preprocessing import Preprocessing

train_path = 'dataset/autoencoder/train_test.csv'
test_path = 'dataset/autoencoder/train_test.csv'
validation = 0.8
batch_size = 32

a = Preprocessing()
a.setdata(train_path, test_path, validation)
train_data, test_data, validation = a.autoencoder()

train_batch = mx.gluon.data.DataLoader(train_data, batch_size, shuffle=False)
val_batch = mx.gluon.data.DataLoader(validation, batch_size, shuffle=False)

for d in train_batch:
    break


model = mx.gluon.nn.Sequential()

with model.name_scope():
    model.add(mx.gluon.rnn.LSTM(1))
    model.add(mx.gluon.nn.Dense(1, activation='tanh'))

L = gluon.loss.L2Loss()

def evaluate_accuracy(data_iterator, model, L):
    loss_avg = 0.
    for i, data in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 1, 1))
        output = model(data)
        loss = L(output, data)
        loss_avg = (loss_avg * i + nd.mean(loss).asscalar()) / (i + 1)
    return loss_avg

ctx = mx.cpu()

model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# 이 모델에는 sgd 최적화 함수, 학습률은 0.01이 가장 최적으로 보임
# sgd에 비해 너무 작지 않고, 최적화에 시간이 오래 걸리지 않으며 너무 크지 않는다
# 그래서 loss function의 최소값을 넘지 않는다
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


### Let’s run the training loop and plot MSEs
epochs = 30
training_mse = []        # 평균 제곱 오차를 기록
validation_mse = []

for epoch in range(epochs):
    print(str(epoch+1))
    for i, data in enumerate(train_batch):
        data = data.as_in_context(ctx).reshape((-1, 1, 1))

        with autograd.record():
            output = model(data)
            loss = L(output, data)

        loss.backward()
        trainer.step(batch_size)

    train_error = evaluate_accuracy(train_batch, model, L)
    val_error = evaluate_accuracy(val_batch, model, L)

    training_mse.append(train_error)
    validation_mse.append(val_error)

    print('Epoch %d. training_mse: %.3f, validation_mse: %.3f' % (epoch, train_error, val_error))

# print(
#             "Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec"
#             % (epoch, train_loss / n, train_acc / m, test_acc, time() - start))

plt.plot(training_mse, color='r')
plt.plot(validation_mse, color='b')

print(training_mse[-1], validation_mse[-1])

len(training_mse)