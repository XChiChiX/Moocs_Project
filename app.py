from flask import Flask, request, send_file
from flask import render_template
import os

from voiceToText import generateTextFile

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

        generateTextFile(file_path)

        os.remove(file_path)

        return {'message': '上傳成功', 'file_path': file_path}
    else:
        return {'message': '未選擇檔案'}

# @app.route('/api/download/textOnly.txt', methods=['POST'])
# def generate_text_file():
#     file_path = os.path.join(os.getcwd(), "textOnly.txt")

#     # 返回文件
#     return send_file(file_path, as_attachment=True)

@app.route('/api/download/<fileName>', methods=['POST'])
def generate_text_file(fileName):
    file_path = os.path.join(os.getcwd(), fileName)

    # 返回文件
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.debug = True
    app.run()