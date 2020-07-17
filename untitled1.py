import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from gensim.models import FastText
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support as score
import fasttext
import time
import visen
from pyvi import ViTokenizer

colnames = ['id', 'free_text']
data1 = pd.read_csv('data/02_train_text.csv', names=colnames)

colnames2 = ['id', 'label_id']
data2 = pd.read_csv('data/03_train_label.csv')
df = pd.merge(data1, data2, on="id", how="inner")
del df['id']

# Divide dataset to 8/2 train, test for label 0,1,2
Df1 = pd.DataFrame()
Df1['text_0'] = df['free_text'].groupby(df['label_id']).apply(list)[0]
np.shape(Df1['text_0'])
Df1['label_0'] = np.zeros((18614,), dtype=np.int32)

Df2 = pd.DataFrame()
Df2['text_1'] = df['free_text'].groupby(df['label_id']).apply(list)[1]
np.shape(Df2['text_1'])
Df2['label_1'] = np.ones((1022,), dtype=np.int32)

Df3 = pd.DataFrame()
Df3['text_2'] = df['free_text'].groupby(df['label_id']).apply(list)[2]
Df3['label_2'] = np.full(shape=709, fill_value=2, dtype=np.int32)

from sklearn.model_selection import train_test_split

x_data_0 = Df1['text_0']
y_data_0 = Df1['label_0']
x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(np.array(x_data_0), np.array(y_data_0), test_size=0.2,
                                                            shuffle=False)

x_data_1 = Df2['text_1']
y_data_1 = Df2['label_1']
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(np.array(x_data_1), np.array(y_data_1), test_size=0.2,
                                                            shuffle=False)

x_data_2 = Df3['text_2']
y_data_2 = Df3['label_2']
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(np.array(x_data_2), np.array(y_data_2), test_size=0.2,
                                                            shuffle=False)

# get data before divide dataset
X_train = np.concatenate((x_train_0, x_train_1, x_train_2), axis=None)
X_test = np.concatenate((x_test_0, x_test_1, x_test_2), axis=None)
Y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=None)
Y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=None)

X_train = [each_string.lower() for each_string in X_train]
X_test = [each_string.lower() for each_string in X_test]

abbrevs = {r'\bk\b': 'không', r'\bko\b': 'không', r"\b's\b": ' ', r"\be\b": 'em', r"\bsđt\b": 'điện thoại',
           r"\bib\b": 'inbox', r"\bctv\b": 'cộng tác viên',r"\bđkm\b": 'dm',r"\bvcl\b": 'vl',r"\bvclin\b": 'vl',r"\bđm\b": 'dm',r"\bd.m\b": 'dm',r"\bdmm\b": 'dm',
           r"\bhnoi\b": 'hà nội',r"\bhanoi\b": 'hà nội',r"\bfb\b": 'facebook',r"\bcmt\b": 'comment',r"\bcoment\b": 'comment',r"\bstt\b": 'status'}


def replace_all(text, dic):
    for i, j in dic.items():
        text = re.sub(i, j, text)
    return text


def normalize_text(sentences):
    normalize = []
    for sent in sentences:
        normalize.append(replace_all(sent, abbrevs))
    return normalize


first_clean = normalize_text(X_train)
first_clean_test = normalize_text(X_test)


def cleanup_text(sentences):
    cleaned_text = []
    for sent in sentences:
        # get hastag
        sent = re.sub(r'(?:^|\s)(\#\w+)', ' hastag', sent)
        # get email
        sent = re.sub(r'[\w\.-]+@[\w\.-]+', 'email', sent)
        # get url
        sent = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                      ' url', sent)
        # dupplicate word
        # sent = re.sub(r'(\w)\1+', r'\1', sent)
        # delete string that contain number
        sent = re.sub(r"\w*\d\w*", ' ', sent)

        # y = re.sub(r'\b\d+([\.,]\d+)?', ' number', g)
        # delete number in string
        # g = re.sub(r"\d*([^\d\W]+)\d*", r'\1', z)
        sent = re.sub('[^\w ]', ' ', sent)
        sent = visen.clean_tone(sent)
        cleaned_text.append(sent)
    return cleaned_text


second_clean = cleanup_text(first_clean)
second_clean_test = cleanup_text(first_clean_test)


# tokenizer for sentencse
def tokenize_sentences(sentences):
    tokens_list = []
    for sent in sentences:
        tokens = ViTokenizer.tokenize(sent)
        tokens_list.append(tokens)
    return tokens_list


third_clean = tokenize_sentences(second_clean)
third_clean_test = tokenize_sentences(second_clean_test)
third_clean_test_set = np.concatenate((third_clean, third_clean_test), axis=None)
# np.savetxt('data/test1.txt', third_clean_test_set, fmt="%s")

def split_word(sentences):
    new_arr = []
    for sent in sentences:
        new_arr.append(sent.split())
    return new_arr


fourth_clean = split_word(third_clean)
fourth_clean_test = split_word(third_clean_test)
fourth_clean_test_set = split_word(third_clean_test_set)

# model = fasttext.train_supervised('data/test1.txt', minn=2, maxn=7, dim=200)
# model.save_model("models/fasttext_1.bin")
# model = fasttext.load_model("models/fasttext_1.bin")
# print(model.get_nearest_neighbors('ô_nhiễm'))
# model.words
def get_train_input(train_texts):
    train_input = []
    model = fasttext.load_model("models/fasttext_1.bin")
    for line in train_texts:
        embedding = []
        for x in range(80):
            if len(line) <= x:
                embedding.append(np.zeros(200, dtype=np.float32))
            else:
                # try:
                c = model.get_word_vector(line[x])
                # except KeyError:
                #     c = np.zeros(80, dtype=np.float32)
                embedding.append(c)
        train_input.append(embedding)
    return train_input

start_time = time.time()
X_text = np.array(get_train_input(fourth_clean))
X_text_test = np.array(get_train_input(fourth_clean_test))

X_label = tf.keras.utils.to_categorical(Y_train, num_classes=3)

input_dim = (X_text.shape[1], X_text.shape[2])

from keras import backend as K
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# build model
model = Sequential()
model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=input_dim))
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(32)))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
model.summary()
history = model.fit(X_text, X_label, epochs=50, batch_size=64,verbose=1)

print()
print("Execution Time %s seconds: " % (time.time() - start_time))
from sklearn.metrics import classification_report

start_time_test = time.time()
prediction = model.predict(X_text_test)
label = []
for n in prediction:
    label.append(np.argmax(n))
# print(prediction)

print(classification_report(Y_test, label))
print()
print("Execution Time test %s seconds: " % (time.time() - start_time_test))
