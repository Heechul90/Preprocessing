import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical


class Preprocessing():
    def setdata(self, data_path, val):
        self.data_path = data_path
        self.val = val

    def text_generation(self):
        path, validation = self.data_path, self.val
        data = pd.read_csv(path, header=None)

        text = []
        text.extend(list(data[0].values))

        def punctu(s):
            # s = s.encode('utf-8').decode('ascii', 'ignore')
            return ''.join(c for c in s if c not in punctuation).lower()

        text = [punctu(x) for x in text]

        t = Tokenizer()
        t.fit_on_texts(text)
        vocab_size = len(t.word_index) + 1

        sequences = list()

        for line in text:
            encoded = t.texts_to_sequences([line])[0]
            for i in range(1, len(encoded)):
                sequence = encoded[:i+1]
                sequences.append(sequence)

        max_len = max(len(l) for l in sequences)
        sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
        sequences = np.array(sequences)
        X = sequences[:,:-1]
        y = sequences[:,-1]

        y = to_categorical(y, num_classes=vocab_size)

        train_data = X[:int(len(X) * validation)]
        validation_data = X[int(len(X) * validation):]
        train_label = y[:int(len(y) * validation)]
        validation_label = y[int(len(y) * validation):]

        return train_data, validation_data, train_label, validation_label


