import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mxnet as mx
import gluonnlp as nlp

data = pd.read_csv('dataset/ArticlesApril2018.csv')

data.columns
data.loc[0]

for i in range(len(data['headline'])):
    if data['headline'][i] == 'Unknown':
        data.drop(i, inplace=True)



len(data)
data = data['headline']
data.to_csv('dataset/text_test.csv', index=None, sep=',', encoding='utf-8')





########################################################################################################################
import pandas as pd
text = pd.read_csv('dataset/text_test.csv', sep=',', encoding='utf-8')

print('총 샘플의 개수 : {}'.format(len(text)))             # 현재 샘플의 개수

text = text.values

def repreprocessing(s):
    s = s
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화

text = [repreprocessing(x) for x in text]
text[:5]


t = Tokenizer()
t.fit_on_texts(text)                      # t.fit_on_texts = 단어 인텍스를 구축
vocab_size = len(t.word_index) + 1        # t.word_index = 구축된 단어와 인덱스를 튜플로 반환
print('단어 집합의 크기 : %d' % vocab_size)


# 총 3,494개의 단어가 존재합니다.
# 정수 인코딩과 동시에 하나의 문장을 여러 줄로 분해하여 훈련 데이터를 구성
sequences = list()

for line in text:                              # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
    encoded = t.texts_to_sequences([line])[0]  # 문자열을 정수 인덱스로 변환
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

sequences[:11]                                 # 11개의 샘플 출력
len(sequences)

# 어떤 정수가 어떤 단어를 의미하는지 알아보기 위해 인덱스로부터 단어를 찾는 index_to_word를 만듭니다.
index_to_word={}
for key, value in t.word_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

print('빈도수 상위 582번 단어 : {}'.format(index_to_word[1]))

# y 데이터를 분리하기 전에 전체 샘플의 길이를 동일하게 만드는 패딩 작업을 수행합니다.
# 패딩 작업을 수행하기 전에 가장 긴 샘플의 길이를 확인합니다.
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))

# 가장 긴 샘플의 길이인 24로 모든 샘플의 길이를 패딩
sequences = pad_sequences(sequences,
                          maxlen=max_len,  # 최대 길이를 설정
                          padding='pre')   # pre: 앞으로 값을 채움, post: 뒤로 값을 채움
print(sequences[:3])

# 맨 우측 단어만 레이블로 분리
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X[:3])
print(y[:5]) # 레이블

# 레이블 데이터 y에 대해서 원-핫 인코딩을 수행
y = to_categorical(y, num_classes=vocab_size)
print(y[:5])

### 1.2 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM


### 각 단어의 임베딩 벡터는 10차원을 가지고, 128의 은닉 상태 크기를 가지는 LSTM을 사용
# Embedding() : Embedding()은 단어를 밀집 벡터로 만드는 역할을 합니다.
# 인공 신경망 용어로는 임베딩 층(embedding layer)을 만드는 역할을 합니다.
# Embedding()은 정수 인코딩이 된 단어들을 입력을 받아서 임베딩을 수행합니다.
# -	        원-핫 벡터	            임베딩 벡터
# 차원	    고차원(단어 집합의 크기)	저차원
# 다른 표현	희소 벡터의 일종	        밀집 벡터의 일종
# 표현 방법	수동	                    훈련 데이터로부터 학습함
# 값의 타입	1과 0	                실수
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))   # y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

print("\n Test Accuracy: %.4f" % (model.evaluate(X, y)[1]))   # Test Accuracy: 0.9246

# sentence_generation을 만들어서 문장을 생성
def sentence_generation(model, t, current_word, n):                  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word                                         # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n):                                               # n번 반복
        encoded = t.texts_to_sequences([current_word])[0]            # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)

    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items():
            if index == result:                                      # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break                                                # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word                    # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word                             # 예측 단어를 문장에 저장

    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

# 임의의 단어 'i'에 대해서 10개의 단어를 추가 생성
print(sentence_generation(model, t, 'i', 10))

# 임의의 단어 'how'에 대해서 10개의 단어를 추가 생성
print(sentence_generation(model, t, 'how', 10))