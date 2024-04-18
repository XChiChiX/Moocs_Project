from flask import Flask, request, send_file
from flask import render_template
import os

from pyScripts import generateTextFile
from pyScripts import claude3api
from pyScripts import split_txt
from pyScripts import splitVideo
from pyScripts import zip_mp4_files
# from VoiceClone import VoiceClone
# import CloneVoiceStamps
# from LipSync import create_video_for_summary_and_questions, upload_files_to_s3, synclabs_api, delete_files_from_s3, add_subtitles, concat_results

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
        # uploaded_file.save(file_path)
        # generateTextFile(file_path)

        # claude3api()
        # split_txt()
        # splitVideo.split_video(uploaded_file.filename)
        # VoiceClone()
        # CloneVoiceStamps.start()
        # create_video_for_summary_and_questions()
        # upload_files_to_s3()
        # synclabs_api()
        # delete_files_from_s3()
        # add_subtitles()
        # concat_results()

        # os.remove(file_path)

        zip_mp4_files()

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