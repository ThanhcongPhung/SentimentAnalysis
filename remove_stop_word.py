from nltk.corpus import stopwords
import re

text = """Trang Bằng shared a post to the group: MUA BÁN ONLINE TẠI HÀ TĨNH. <URL> Trang .
 CÓ NHỮNG CÔ GÁI - LÔNG RẬM HƠN CON TRAI 😂😂😂😂ngại quá nhỉ 🎯.
.
Đừng lo : #đã #có Wax veo giải quyết hết vấn đề của bạn. 💢💢💢.
‼Sạch bong lớp lông lá, trả lại cho bạn làn da mịn màng, se khít lỗ chân lông, hết viêm da, lông hết mọc ngược. .
Tha hồ dơ tay đánh khẽ nhé .
‼Kết hợp bôi mỡ trăn 👉 giúp bạn triệt lông vĩnh viễn. .
⛔Inbox ngay cho mình nha.
☎☎ 0358618XXX"""


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
#
#
# print(cleanup_text(text))

# test_str = text.replace('\n', '')
# reviews_train_clean = re.sub(r"[^-/().&' \w]|_", "", test_str)
#
# reviews_train_test = re.sub(r"[.;:!\'?,\"()\[\]]", "", reviews_train_clean)
# reviews_train_final = re.sub(r'[^\w]', ' ', reviews_train_test)
#
# print(reviews_train_final.lower())
#
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

    return removed_stop_words

# def remove_stop_words(corpus):
#     removed_stop_words = []
#     # for review in corpus:
#     removed_stop_words.append(
#             ' '.join([word for word in corpus.split()
#                       if word not in stop_word])
#         )
#     return removed_stop_words
# removed_stop_word_vietnam = remove_stop_words(reviews_train_final.lower())
# bowA= removed_stop_word_vietnam[0].split(' ')
# word_dict=set(bowA).union()
