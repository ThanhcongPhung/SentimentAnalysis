import re
import visen

s = """Mua Sắm Làm Đẹp - muasamlamdep.Com <URL> CĂN HỘ GREEN TOWN BÌNH TÂN TPHCM .
 CHÍNH CHỦ CẦN SANG NHƯỢNG GẤP CĂN 49.02M2 ĐẸP NHƯ HÌNH.
Giá: 1.4 tỷ bao hết thuế phí.
Lầu 8 khối B3 đang đi vào hoàn thiện bành giao 2019.
Phòng khách,2 phòng ngủ lót sàn 200gram gỗ vcc 09.75000XXXX xefxb8x8f caoooooooo cấp HQ,phòng bếp,nhà vệ sinh,2 logia ban công....
Chi tiết dự án: www.greentown.com.vn"""
new_string = re.sub(r'(?:^|\s)(\#\w+)', ' hastag', s)
email = re.sub(r'[\w\.-]+@[\w\.-]+', 'email', new_string)
x = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', ' url',
           email)
y = re.sub(r'\b\d+([\.,]\d+)?', ' number', x)
z = re.sub(r'(\w)\1+', r'\1', y)
g = re.sub(r"\d*([^\d\W]+)\d*", r'\1', z)
# h = re.sub(r'[^a-zA-Z0-9]+', '', g)
print(g)
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