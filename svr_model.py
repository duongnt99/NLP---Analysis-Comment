from flask import Flask, render_template
from flask_cors import CORS, cross_origin
from flask import request, jsonify

import sys
import joblib

sys.modules['sklearn.externals.joblib'] = joblib
import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# Hàm load dữ liệu URL và trả về các comment + hình ảnh sản phẩm
def load_url_selenium(url):
    # try:

        # Cài đặt tham số Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chome_path = os.path.join("driver", "chromedriver.exe")
        driver = webdriver.Chrome(executable_path=os.path.abspath(chome_path),   chrome_options=chrome_options)

        # Mở URL
        driver.get(url)
        time.sleep(1)
        # Cuộn xuống cuối trang để đảm bảo trang load hết
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)

        # Lấy danh sách comment
        review_item = driver.find_elements_by_class_name("review-comment__content")
        print('-----------------------------------------------\n',review_item)
        csvdata = []

        # Lặp và thêm vào list
        for item in review_item:
            print('*********************\n', item)
            content = item.text
            csvdata.append([content])

        # Lấy hình ảnh sản phẩm
        image = driver.find_element_by_xpath("/html/body/div[1]/div[1]/main/div[4]/div/div[1]/div[1]/div[1]/div/div/div/div/img")
        image = image.get_attribute("src")

        # Trả về danh sách comment +ảnh sản phẩm
        return csvdata, image
    # except:
    #     return None, None



# Hàm chuẩn hoá dữ liệu
def standardize_data(row):
    # remove stopword

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row


# Hàm tách từ
def tokenizer(row):
    return word_tokenize(row, format="text")


# Hàm phân tích kết quả trả về của model
def analyze(result):
    # Đếm số comment xấu
    bad = np.count_nonzero(result)
    # Tính số comment tốt
    good = len(result) - bad

    # Nếu tốt nhiều hơn xấu thì trả về tốt
    if good > bad:
        return "Sản phẩm này tốt. Có thể mua! ", good / (good + bad)
    else:
        # Ngược lại trả về xấu
        return "Sản phẩm này không tốt. Bạn hãy cân nhắc! ", good / (good + bad)


# Load model và dữ liệu TF-IDF
emb = joblib.load('data/tfidf.pkl')
model = joblib.load('data/saved_model.pkl')

# ######## Khởi tạo webserver
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def home():
    # Trả về giao diện
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin(origin='*')
def predict():
    # Lấy thông tin URL từ client gửi lên
    url = request.json['url']
    url = url.strip()

    # Load dữ liệu từ tiki
    data, image_url = load_url_selenium(url)

    # Nêu có dữ liệu trả về
    if data is not None:
        if len(data) > 0:
            # Chuẩn hoá
            data_frame = pd.DataFrame(data)
            data_frame[0] = data_frame[0].apply(standardize_data)

            # Tách từ
            data_frame[0] = data_frame[0].apply(tokenizer)

            # Chuyển comment thành vector
            X_val = data_frame[0]
            X_val = emb.transform(X_val)

            # Dự đoán và xử lý
            result, cp = analyze(model.predict(X_val))

            # Trả về cho client
            comment_list = "** DANH MỤC CÁC COMMENTS: **\n --------------------------------------\n"
            for i in data:
                print(i)
                comment_list += "- " + str(i[0]).replace("\n", "").replace("<br>", "") + "\n"

            dict = {
                "cl": str(result),
                "cp": "%.2f" % (cp * 100),
                "img": image_url,
                "comments": comment_list
            }
        else:
            dict = {
                "cl": "Sản phẩm hiện tại không có bình luận nào!",
                "cp": "",
                "img": image_url
            }
    else:
        dict = {
            "cl": "Địa chỉ sản phẩm không hợp lệ!",
            "cp": ""
        }
    # Trả về kết quả
    print("Trả về kết quả cho client: ", dict)
    return jsonify(dict)


# Bật Backend ở cổng 9999
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9999')
