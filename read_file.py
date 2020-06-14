import pandas as pd
from collections import Counter

from keras import Input
from keras.layers import Reshape, LSTM, Dense

from dict_models import LongMatchingTokenizer
from remove_stop_word import cleanup_text, remove_stop_words
from nltk import ngrams
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

from text_classification import BiDirectionalLSTMClassifier

colnames = ['id', 'free_text']
data1 = pd.read_csv('data/02_train_text.csv', names=colnames)
free_text = data1.free_text.tolist()
x = [element.lower() for element in free_text]
length = len(x)
# # print(length)
clean_punction = cleanup_text(x[1:length])
# print(clean_punction)
lm_tokenizer = LongMatchingTokenizer()
clean_stopword = remove_stop_words(clean_punction)
# for x in range(0, 1):
#     # print(Counter(clean_stopword[x].split(' ')).most_common())
#     print(clean_stopword[x])
#     tokens = lm_tokenizer.tokenize(clean_stopword[x])
#     print(tokens)
#     model = Word2Vec([tokens], size=10, window=5, min_count=1, workers=4)
#     word = ["trứng"]
#     vector = model.wv[word]
#     print(vector)
#     sim_words = model.wv.most_similar(word)
#     print(sim_words)
# print(clean_stopword[1])
tokens = lm_tokenizer.tokenize(clean_stopword[2])
print(tokens)
model = Word2Vec([tokens], size=100, window=2, min_count=1, workers=4)
word = ["lol"]
vector = model.wv[word]
print(vector)
sim_words = model.wv.most_similar(word)
print(sim_words)


# new_corpus = ''.join(map(str, clean_stopword))


# word = "điiiiii"
# vector = model.wv[word]
# print(vector)
# sim_words = model.wv.most_similar(word)
# print(sim_words)

# colnames2 = ['id', 'label_id']
# data2 = pd.read_csv('03_train_label.csv')
# df = pd.merge(data1, data2, on="id", how="inner")
# print(df)
