##### Test

### 함수 import
import mxnet as mx
import gluonnlp
from mxnet import gluon, autograd, nd



### Examples
text_data = " hello world \\n hello nice world \\n hi world \\n"

counter = gluonnlp.data.count_tokens(text_data)
my_vocab = gluonnlp.Vocab(counter)
fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple.vec')
my_vocab.set_embedding(fasttext)
my_vocab.embedding[['hello', 'world']]

my_vocab[['hello', 'world']]

input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(my_vocab.embedding.idx_to_vec)
layer(nd.array([5, 4]))

glove = gluonnlp.embedding.create('glove', source='glove.6B.50d.txt')
my_vocab.set_embedding(glove)
my_vocab.embedding[['hello', 'world']]
