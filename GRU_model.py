import pandas as pd
import re
import numpy as np
from dict_models import LongMatchingTokenizer
from gensim.models import FastText
from gensim.models import KeyedVectors
import fasttext
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from gensim.models import Word2Vec
import time

# read file train
colnames = ['id', 'free_text']
data1 = pd.read_csv('data/02_train_text.csv', names=colnames)

colnames2 = ['id', 'label_id']
data2 = pd.read_csv('data/03_train_label.csv')
df = pd.merge(data1, data2, on="id", how="inner")
del df['id']

# read file test
# colnames3 = ['id', 'free_text']
data3 = pd.read_csv('data/04_test_text.csv')
test_file = data3.free_text.tolist()

test_file = [each_string.lower() for each_string in test_file]
# print(data3.head())
data4 = pd.read_csv('data/01_label_mapping.csv')


# lm_tokenizer = LongMatchingTokenizer()
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def cleanup_text(text_str):
    cleaned_text = []
    for text in text_str:
        text = re.sub(r"[^-/().&' \w]|_", "", text)
        text = re.sub(r"[;:!\'?,\"()\[\]]", "", text)
        text = re.sub(r'[^\w]', ' ', text)
        # remove punctuation
        text = re.sub('[!#?,.:";]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        cleaned_text.append(text)
    return cleaned_text


with open("vietnamese-stopwords.txt", encoding="utf8") as f:
    stop_word = f.readlines()
stop_word = [x.strip() for x in stop_word]


def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in stop_word])
        )

    return cleanup_text(removed_stop_words)


df['free_text'] = remove_stop_words(df['free_text'].str.lower())
train = np.array(df['free_text'])
test_file_remove_list = remove_stop_words(test_file)


def tokenize_sentences(sentences):
    """
    Tokenize or word segment sentences
    :param sentences: input sentences
    :return: tokenized sentence
    """
    tokens_list = []
    lm_tokenizer = LongMatchingTokenizer()
    for sent in sentences:
        tokens = lm_tokenizer.tokenize(sent)
        tokens_list.append(tokens)
    return tokens_list


tokenize = tokenize_sentences(df['free_text'])

test_tokenize = tokenize_sentences(test_file_remove_list)


def read_data(sentences):
    embed_sentences = []
    for sent in sentences:
        for word in sent:
            embed_sentences.append(word)
    return embed_sentences


def get_train_input(train_texts):
    train_input = []
    model = KeyedVectors.load("models/word2vec_skipgram.model")
    for line in train_texts:
        embedding = []
        for x in range(140):
            if len(line) <= x:
                embedding.append(np.zeros(100, dtype=np.float32))
            else:
                embedding.append(model.wv[line[x]])
        train_input.append(embedding)
    return train_input


if __name__ == '__main__':
    # model_fasttext = FastText(size=100, window=5, min_count=2, workers=4, sg=1)
    # model_fasttext.build_vocab(train)
    # model_fasttext.train(train, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
    #
    # model_fasttext.wv.save("models/fasttext_gensim.model")
    model = Word2Vec(tokenize, size=100, window=3, min_count=1, workers=4)
    model.save("models/word2vec_skipgram.model")
    start_time = time.time()

    # X_test = np.reshape(np.array(test_tokenize), np.array(test_tokenize).shape + (1,1))
    Y_train = np.array(df['label_id'])
    # Y_test = np.array(data4)
    X_train = np.array(get_train_input(tokenize))
    print(np.shape(X_train))
    # print(np.shape(X_test))
    print(np.shape(Y_train))
    # print(np.shape(Y_test))
    input_dim = (X_train.shape[1], X_train.shape[2])

    model = Sequential()

    model.add(GRU(64, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=15, batch_size=64)
    scores = model.evaluate(X_train, Y_train)
    model.save("models/GRU_model1.model")
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print()
    print("Execution Time %s seconds: " % (time.time() - start_time))
