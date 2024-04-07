# pip install moviepy
# pip install pydub
# pip install elevenlabs
# typing-extensions==4.7.1
# https://elevenlabs.io/docs/api-reference/text-to-speech

# max 25 voice sample
import os
import requests
from elevenlabs import clone, generate, play, save
from elevenlabs import set_api_key
from moviepy.editor import *
from pydub import AudioSegment
import math
import random

def VoiceCopy(numbers , key):

    VoiceId = ""
    VoiceName = "Test1"
    VoiceDescription = "This is test"
    VoiceFiles = []

    mp3_input_path = "./data/audio/source.mp3"
    mp3_output_folder = "./data/output/"
    target_min = 8
    segment_duration_ms = target_min * 60 * 1000

    set_api_key(ElevenLabsAPI)

    print("Program Start !")

    # mp4 to mp3

    video = VideoFileClip("./data/video/source.mp4")
    video.audio.write_audiofile("./data/audio/source.mp3")

    print("mp4 to mp3 successful !")

    # cut mp3

    audio = AudioSegment.from_mp3(mp3_input_path)

    # 設定每個檔案的持續時間（以毫秒為單位）
    segment_duration = segment_duration_ms

    # 取得輸入MP3檔案的名稱（不含副檔名）
    file_name = os.path.splitext(os.path.basename(mp3_input_path))[0]

    # 計算分割的片段數量
    num_segments = len(audio) // segment_duration + 1

    # 切割並儲存每個片段
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[start_time:end_time]

        # 確保最後一個片段不會超過總長度
        if i == num_segments - 1:
            segment = audio[start_time:]

        # 輸出檔案路徑
        output_path = os.path.join(mp3_output_folder, f"sample{i + 1}.mp3")

        # 儲存片段
        segment.export(output_path, format="mp3")

    print("cut mp3 successful !")

    # get voice smaple

    files = os.listdir(mp3_output_folder)
    filtered_files = [filename for filename in files if filename.startswith("sample") and filename.endswith(".mp3")]

    for filename in os.listdir(mp3_output_folder):

        if len(filtered_files) <= 25:
            if filename.startswith("sample") and filename.endswith(".mp3"):
                # 構建文件的完整路徑
                VoiceFiles.append(os.path.join(mp3_output_folder, filename))
        else:
            # 从符合条件的文件中随机选择 25 个
            VoiceFiles.append(random.sample(filtered_files, 25))

    print("get voice successful !")

    # make copy voice
    voice = clone(
        name=VoiceName,
        description=VoiceDescription, # Optional
        files=VoiceFiles
    )

    for i in range(1 , numbers +1 ):

        print(str(i)+" : \n")

        with open('./data/clips/question_part_'+str(i)+'.txt', 'r', encoding='utf-8') as q_file:
            question = q_file.readlines()

        with open('./data/clips/summary_part_'+str(i)+'.txt', 'r', encoding='utf-8') as s_file:
            summary = s_file.readlines()

        question = [line.strip() for line in question]
        summary = [line.strip() for line in summary]
        summary_str = "".join(summary)
        question_str = "".join(question)

        audio = generate(text=summary_str, voice=voice,model="eleven_multilingual_v2")

        save(audio,"./data/ans/Summary"+str(i)+".mp3")
        print("Summary "+str(i)+" successful !")

        audio = generate(text=question_str, voice=voice,model="eleven_multilingual_v2")

        save(audio,"./data/ans/Question"+str(i)+".mp3")
        print("Question "+str(i)+" successful !")

    print("copy voice successful !")

    # delete

    delete_url = "https://api.elevenlabs.io/v1/voices"

    headers = {
    "Accept": "application/json",
    "xi-api-key": ElevenLabsAPI
    }

    response = requests.get(delete_url, headers=headers)

    data = response.json()

    for target_voice in data['voices']:
        if target_voice['name'] == VoiceName:
            VoiceId = target_voice['voice_id']

    url = "https://api.elevenlabs.io/v1/voices/"+VoiceId

    response = requests.delete(url, headers=headers)

    if(response.text == "{\"status\":\"ok\"}"):
        print("Delete Successful !")
    else:
        print("Something Error !")
        print(response.text)


ElevenLabsAPI = ""

VoiceCopy(3,ElevenLabsAPI)