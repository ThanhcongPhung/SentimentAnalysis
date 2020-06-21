import numpy as np
import pandas as pd
from tensorflow import keras
import re

from dict_models import LongMatchingTokenizer

model = keras.models.load_model("models/GRU_model.model")
model.summary()
# data3 = pd.read_csv('data/04_test_text.csv')
# test_file = data3.free_text.tolist()
# test_file = [each_string.lower() for each_string in test_file]


# def cleanup_text(text_str):
#     cleaned_text = []
#     for text in text_str:
#         text = re.sub(r"[^-/().&' \w]|_", "", text)
#         text = re.sub(r"[;:!\'?,\"()\[\]]", "", text)
#         text = re.sub(r'[^\w]', ' ', text)
#         # remove punctuation
#         text = re.sub('[!#?,.:";]', ' ', text)
#         # remove multiple spaces
#         text = re.sub(r' +', ' ', text)
#         # remove newline
#         text = re.sub(r'\n', ' ', text)
#         cleaned_text.append(text)
#     return cleaned_text
#
#
# with open("vietnamese-stopwords.txt", encoding="utf8") as f:
#     stop_word = f.readlines()
# stop_word = [x.strip() for x in stop_word]
#
#
# def remove_stop_words(corpus):
#     removed_stop_words = []
#     for review in corpus:
#         removed_stop_words.append(
#             ' '.join([word for word in review.split()
#                       if word not in stop_word])
#         )
#
#     return cleanup_text(removed_stop_words)
#
#
# test_1 = remove_stop_words(test_file)
#
#
# def tokenize_sentences(sentences):
#     """
#     Tokenize or word segment sentences
#     :param sentences: input sentences
#     :return: tokenized sentence
#     """
#     tokens_list = []
#     lm_tokenizer = LongMatchingTokenizer()
#     for sent in sentences:
#         tokens = lm_tokenizer.tokenize(sent)
#         tokens_list.append(tokens)
#     return tokens_list
#
#
# X_test = tokenize_sentences(test_1)
# print(X_test[1])
# np.testing.assert_allclose(
#   model.predict(X_test[1]))
#
