import mxnet as mx
import gluonnlp as nlp
ctx = mx.cpu()

lm_model, vocab = nlp.model.get_model(name = 'awd_lstm_lm_1150',
                                      dataset_name = 'wikitext-2',
                                      pretrained = True,
                                      ctx = ctx)


### Beam Search Sampler
scorer = nlp.model.BeamSearchScorer(alpha=0, K=5, from_logits=False)

class LMDecoder(object):
    def __init__(self, model):
        self._model = model
    def __call__(self, inputs, states):
        outputs, states = self._model(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states
    def state_info(self, *arg, **kwargs):
        return self._model.state_info(*arg, **kwargs)
decoder = LMDecoder(lm_model)


eos_id = vocab['.']
sampler = nlp.model.BeamSearchSampler(beam_size=4,
                                      decoder=decoder,
                                      eos_id=eos_id,
                                      scorer=scorer,
                                      max_length=20)


### Generate Sequences with Beam Search

bos = 'I love it'.split()
bos_ids = [vocab[ele] for ele in bos]
begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
if len(bos_ids) > 1:
    _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
                               begin_states)
inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])

# samples have shape (1, beam_size, length), scores have shape (1, beam_size)
samples, scores, valid_lengths = sampler(inputs, begin_states)

samples = samples[0].asnumpy()
scores = scores[0].asnumpy()
valid_lengths = valid_lengths[0].asnumpy()
print('Generation Result:')
for i in range(3):
    sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
    print([' '.join(sentence), scores[i]])


### Sequence Sampler
sampler = nlp.model.SequenceSampler(beam_size=4,
                                    decoder=decoder,
                                    eos_id=eos_id,
                                    max_length=100,
                                    temperature=0.97)


bos = 'I love it'.split()
bos_ids = [vocab[ele] for ele in bos]
begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
if len(bos_ids) > 1:
    _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
                               begin_states)
inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
samples, scores, valid_lengths = sampler(inputs, begin_states)
samples = samples[0].asnumpy()
scores = scores[0].asnumpy()
valid_lengths = valid_lengths[0].asnumpy()
sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[0][:valid_lengths[0]]]
print('Generation Result:')
for i in range(5):
    sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
    print([' '.join(sentence), scores[i]])


########################################################################################################################
import warnings
warnings.filterwarnings('ignore')

import itertools
import time
import math
import logging
import random

import mxnet as mx
import gluonnlp as nlp
import numpy as np
from scipy import stats

nlp.utils.check_version('0.7.0')

context = mx.cpu()  # Enable this to run on CPU
# context = mx.gpu(0)  # Enable this to run on GPU


##### Data
text8 = nlp.data.Text8()
print('# sentences:', len(text8))
for sentence in text8[:3]:
    print('# tokens:', len(sentence), sentence[:5])


counter = nlp.data.count_tokens(itertools.chain.from_iterable(text8))
vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                  bos_token=None, eos_token=None, min_freq=5)
idx_to_counts = [counter[w] for w in vocab.idx_to_token]

def code(sentence):
    return [vocab[token] for token in sentence if token in vocab]

text8 = text8.transform(code, lazy=False)

print('# sentences:', len(text8))
for sentence in text8[:3]:
    print('# tokens:', len(sentence), sentence[:5])


from data import transform_data_fasttext

batch_size=4096
data = nlp.data.SimpleDataStream([text8])  # input is a stream of datasets, here just 1. Allows scaling to larger corpora that don't fit in memory
data, batchify_fn, subword_function = transform_data_fasttext(
    data, vocab, idx_to_counts, cbow=False, ngrams=[3,4,5,6], ngram_buckets=100000, batch_size=batch_size, window_size=5)
