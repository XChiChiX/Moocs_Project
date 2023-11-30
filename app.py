from flask import Flask, request
from flask import render_template
import os

from voiceToText import start

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file:
        # 指定儲存路徑
        save_path = os.getcwd()
        print(save_path)

        # # 路徑不存在則創建
        # os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, uploaded_file.filename)
        print("48763 ", file_path)

        # 儲存檔案
        uploaded_file.save(file_path)

        start(file_path)

        return {'message': '上傳成功', 'file_path': file_path}
    else:
        return {'message': '未選擇檔案'}

if __name__ == '__main__':
    app.debug = True
    app.run()