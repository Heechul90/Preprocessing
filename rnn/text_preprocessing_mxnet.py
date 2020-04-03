# -*- coding: utf-8 -*-


from __future__ import print_function
import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np

path = 'dataset/nlp/timemachine.txt'
ctx = mx.cpu()
seq_length = 64

def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result



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




