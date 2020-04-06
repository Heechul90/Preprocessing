# -*- coding: utf-8 -*-


from __future__ import print_function
import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np
import gluonnlp

path = 'dataset/nlp/timemachine.txt'
ctx = mx.cpu()
seq_length = 64

# time_machine 파일 열기
with open('dataset/nlp/timemachine.txt') as f:
    time_machine = f.read()
len(time_machine)

# 1000개만 test로 사용
time_machine = time_machine[:1000]
len(time_machine)

# 스펠링을 리스트로 바꿈(중복은 같은걸로 처리)
character_list = list(set(time_machine))

vocab_size = len(character_list)

# enumerate로 번호를 매김
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
print(character_dict)

# character_dict을 이용하여 time_machine의 스펠링을 숫자로 바꿈
time_numerical = [character_dict[char] for char in time_machine]
len(time_machine)

# 다시 숫자를 text로 변경하는 코드
print("".join([character_list[idx] for idx in time_numerical[:39]]))

# 1000개의 스펠링을 0과1로 one-hot 인코딩을 함
def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

time_numerical[:2]
print(one_hots(time_numerical[:2]))

# one-hot 인코딩을 다시 text로 바꿈
def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

textify(one_hots(time_numerical[:40]))














def text_generation(path, seq_length):
    with open(path) as f:
        text = f.read()

    character_list = list(set(text))
    vocab_size = len(character_list)

    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e

    time_numerical = [character_dict[char] for char in text]
    len(time_numerical)

    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(time_numerical[:seq_length * num_samples]).reshape((num_samples, seq_length, vocab_size))
    textify(dataset[0])
    len(dataset)




