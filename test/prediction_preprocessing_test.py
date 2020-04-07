from rnn.prediction_preprocessing import Preprocessing
import mxnet as mx
from mxnet import nd, autograd, gluon


path = 'dataset/prediction/test.csv'
seq_length = 14
predict = 7
split = 0.7
batch_size=32

a = Preprocessing()
a.setdata(path, seq_length, predict, split, batch_size)
train_iter, test_iter = a.prediction()

for d, l in train_iter:
    break
d.shape
l.shape

##### model
model = mx.gluon.nn.Sequential()

with model.name_scope():
    model.add(mx.gluon.rnn.LSTM(128))
    model.add(mx.gluon.nn.Dense(1, activation='tanh'))


### Training & Evaluation
# loss 함수 선택
L = gluon.loss.L2Loss() # L2 loss: (실제값 - 예측치)제곱해서 더한 값, L1 loss: (실제값 - 예측치)절대값해서 더한 값

# 평가
def evaluate_accuracy(data_iterator, model, L):
    loss_avg = 0.
    for i, (d, l) in enumerate(data_iterator):
        data = d.as_in_context(ctx).reshape((-1, 1, 1))
        label = l.as_in_context(ctx).reshape((-1, 1, 1))
        output = model(data)
        loss = L(output, data)
        loss_avg = (loss_avg * i + nd.mean(loss).asscalar()) / (i + 1)
    return loss_avg



# cpu or gpu
ctx = mx.cpu()


# Xavier는 모든 layers에서 gradient scale이 거의 동일하게 유지하도록 설계됨
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
    for i, (d, l) in enumerate(train_iter):
        data = d.as_in_context(ctx).reshape((-1, 1, 1))
        label = l.as_in_context(ctx).reshape((-1, 1, 1))

        with autograd.record():
            output = model(data)
            loss = L(output, data)

        loss.backward()
        trainer.step(batch_size)

    training_mse.append(evaluate_accuracy(train_iter, model, L))
    validation_mse.append(evaluate_accuracy(test_iter, model, L))