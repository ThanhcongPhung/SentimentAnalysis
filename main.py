import pandas as pd
import re
import numpy as np
import visen

import remove_stop_word

s = """a lô Mua Sắm Làm Đẹp - muasamlamdep.Com <URL> CĂN HỘ GREEN TOWN BÌNH TÂN TPHCM .
 CHÍNH CHỦ CẦN SANG NHƯỢNG GẤP CĂN 49.02M2 ĐẸP NHƯ HÌNH.
Giá: 1.4 tỷ bao hết thuế phí. k ko không cong's:)))))[]
Lầu 8 khối B3 đang đi vào hoàn thiện bành giao 2019. USA GB
Phòng khách,2 phòng ngủ lót sàn 200gram gỗ vcc 09.75000XXXX xefxb8x8f caoooooooo cấp HQ,phòng bếp,nhà vệ sinh,2 logia ban công....
Chi tiết dự án: www.greentown.com.vn"""

abbrevs = {r'\bk\b': 'không',
           r'\bko\b': 'không',
           'USA': 'United State',
           r"\b's\b": ' '}


def replace_all(text, dic):
    for i, j in dic.items():
        text = re.sub(i, j, text)
    return text


t = replace_all(s, abbrevs)

print(t)
# get hastag
new_string = re.sub(r'(?:^|\s)(\#\w+)', ' hastag', t)
# get email
email = re.sub(r'[\w\.-]+@[\w\.-]+', 'email', new_string)
# get url
x = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', ' url',
           email)
# dupplicate word
z = re.sub(r'(\w)\1+', r'\1', x)
# delete string that contain number
g = re.sub(r"\w*\d\w*", ' ', z)

# y = re.sub(r'\b\d+([\.,]\d+)?', ' number', g)

# delete number in string
# g = re.sub(r"\d*([^\d\W]+)\d*", r'\1', z)
h = re.sub('[^\w ]', ' ', g)
final = visen.clean_tone(h)
remove_stop_word
print(final)
