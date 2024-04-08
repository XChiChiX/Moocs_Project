from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip
import pandas as pd
import os
import timeit
import cv2
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, BertTokenizer, BertModel
import librosa
from pydub import AudioSegment 
import requests
import json
import urllib.request
import boto3
from botocore.exceptions import ClientError
import logging
from transparent_background import Remover
from PIL import Image
import numpy as np
    
class Wav2Vec2BertClassifier(nn.Module):
    """
    Model for deciding the magnitude of movement according to text and audio.
    """
    
    def __init__(self, num_labels):
        super(Wav2Vec2BertClassifier, self).__init__()
        # Wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.freeze_feature_encoder()
        self.layer_weights_wav2vec2 = nn.Parameter(torch.ones(13))
        self.layernorm_wav2vec2 = nn.LayerNorm(768)
        
        # BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.layer_weights_bert = nn.Parameter(torch.ones(12))
        self.layernorm_bert = nn.LayerNorm(768)

        # Late fusion
        self.classifier = nn.Linear(1536, num_labels)
        
    def forward(self, input_wav2vec2, input_bert, mask_bert):
        
        # Wav2vec2 embeddings
        outputs = self.wav2vec2(input_wav2vec2, output_hidden_states=True, return_dict=True)
        layers_output_wav2vec2 = outputs[2]
        layers_output_wav2vec2 = torch.stack(layers_output_wav2vec2, dim=1)
        norm_weights_wav2vec2 = nn.functional.softmax(self.layer_weights_wav2vec2, dim=-1)
        output_wav2vec2 = (layers_output_wav2vec2 * norm_weights_wav2vec2.view(-1, 1, 1)).sum(dim=1)
        output_wav2vec2 = self.layernorm_wav2vec2(output_wav2vec2)
        
        # BERT embeddings
        layers_output_bert = self.bert(input_ids=input_bert, attention_mask=mask_bert,return_dict=True, output_hidden_states=True)
        layers_output_bert = torch.stack(layers_output_bert['hidden_states'][1:])
        norm_weights_bert = nn.functional.softmax(self.layer_weights_bert, dim=-1)
        output_bert = (layers_output_bert * norm_weights_bert.view(-1, 1, 1, 1)).sum(dim=0)
        output_bert = self.layernorm_bert(output_bert)
        
        # Global Average
        output_wav2vec2 = torch.mean(output_wav2vec2, dim=1)
        output_bert = torch.mean(output_bert, dim=1)
        
        # Concatenate two outputs
        output_concat = torch.cat((output_wav2vec2, output_bert), dim=1)
        
        # Generate logits
        logits = self.classifier(output_concat)

        return logits

def create_video_for_summary_and_questions(clip_num):
    """
    生成總結與問題的影片
    
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
        
    檔案需求
    ---------------------
    ./data/ans/Summary{clip_num}.txt
    ./data/ans/Summary{clip_num}.mp3
    ./data/ans/Questions{clip_num}.txt
    ./data/ans/Questions{clip_num}.mp3
    """
    
    print("Start executing concat_clips")
    function_start_time = timeit.default_timer()
    model = Wav2Vec2BertClassifier(num_labels).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoints_path, checkpoint_name))["model_state_dict"])
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    cap = cv2.VideoCapture(os.path.join(concat_source_path, '0.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    if not os.path.exists(os.path.join(fake_audios_path, f"Summary{clip_num}.txt")) or not os.path.exists(os.path.join(fake_audios_path, f"Summary{clip_num}.mp3")) or not os.path.exists(os.path.join(fake_audios_path, f"Questions{clip_num}.txt")) or not os.path.exists(os.path.join(fake_audios_path, f"Questions{clip_num}.mp3")):
        print(f'{clip_num}的總結問題的音檔及文字檔不存在')
        return
    
    texts = pd.read_csv(os.path.join(fake_audios_path, f"Summary{clip_num}.txt"), sep=' ', header=None)
    audio_segment = AudioSegment.from_mp3(os.path.join(fake_audios_path, f"Summary{clip_num}.mp3"))
    
    frame_count = 0
    cap.release()
    
    writer = cv2.VideoWriter(os.path.join(concat_path, f"Summary{clip_num}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for index, (sentence, start_time, end_time) in texts.iterrows():
        
        # Wav2Vec2
        audio_segment[start_time * 1000:end_time * 1000].export(os.path.join(data_path, "temp.mp3"), format="mp3")
        audio_feature = feature_extractor(librosa.load(os.path.join(data_path, "temp.mp3"), sr=16000)[0], return_tensors="pt", padding="max_length", max_length=60000, truncation=True, sampling_rate=16000).to(device)
        
        # BERT
        text_feature = tokenizer(sentence, padding='max_length', max_length=30, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(audio_feature["input_values"], text_feature["input_ids"], text_feature["attention_mask"])
            
        label = torch.argmax(outputs).item()
        print(label)
        
        cap = cv2.VideoCapture(os.path.join(concat_source_path, f'{label}.mp4'))
        
        while cap.isOpened() and frame_count / fps < end_time:
            ret, frame = cap.read()

            if ret is False:
                cap.release()
                cap = cv2.VideoCapture(os.path.join(concat_source_path, '0.mp4'))
                continue
            
            writer.write(frame)
            last_frame = frame
            frame_count += 1
        
        cap.release()
        
    for i in range(int(fps)):
        writer.write(last_frame)
        
    writer.release()
    
    texts = pd.read_csv(os.path.join(fake_audios_path, f"Questions{clip_num}.txt"), sep=' ', header=None)
    audio_segment = AudioSegment.from_mp3(os.path.join(fake_audios_path, f"Questions{clip_num}.mp3"))
    
    frame_count = 0
    cap.release()
    
    writer = cv2.VideoWriter(os.path.join(concat_path, f"Questions{clip_num}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for index, (sentence, start_time, end_time) in texts.iterrows():
        
        # Wav2Vec2
        audio_segment[start_time * 1000:end_time * 1000].export(os.path.join(data_path, "temp.mp3"), format="mp3")
        audio_feature = feature_extractor(librosa.load(os.path.join(data_path, "temp.mp3"), sr=16000)[0], return_tensors="pt", padding="max_length", max_length=60000, truncation=True, sampling_rate=16000).to(device)
        
        # BERT
        text_feature = tokenizer(sentence, padding='max_length', max_length=30, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(audio_feature["input_values"], text_feature["input_ids"], text_feature["attention_mask"])
            
        label = torch.argmax(outputs).item()
        print(label)
        
        cap = cv2.VideoCapture(os.path.join(concat_source_path, f'{label}.mp4'))
        
        while cap.isOpened() and frame_count / fps < end_time:
            ret, frame = cap.read()

            if ret is False:
                cap.release()
                cap = cv2.VideoCapture(os.path.join(concat_source_path, '0.mp4'))
                continue
            
            writer.write(frame)
            last_frame = frame
            frame_count += 1
        
        cap.release()
        
    for i in range(int(fps)):
        writer.write(last_frame)
        
    writer.release()
    
    if os.path.exists(os.path.join(data_path, "temp.mp3")):
        os.remove(os.path.join(data_path, "temp.mp3"))
        
    print("concat_clips cost %d second\n" % (timeit.default_timer() - function_start_time)) 

def upload_files_to_s3(clip_num):
    """
    上傳假影片跟假音檔至AWS S3
    
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
        
    檔案需求
    ---------------------
    ./data/ans/Summary{clip_num}.mp3
    ./data/concat/Summary{clip_num}.mp4
    ./data/ans/Questions{clip_num}.mp3
    ./data/concat/Questions{clip_num}.mp4
    """
    
    print("Start executing upload_files")
    function_start_time = timeit.default_timer()
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(os.path.join(concat_path, f"Summary{clip_num}.mp4")) or not os.path.exists(os.path.join(fake_audios_path, f"Summary{clip_num}.mp3")) or not os.path.exists(os.path.join(concat_path, f"Questions{clip_num}.mp4")) or not os.path.exists(os.path.join(fake_audios_path, f"Questions{clip_num}.mp3")):
        print(f'{clip_num}的總結問題的影片及音檔不存在')
        return
    
    class ObjectWrapper:
        def __init__(self, s3_object):
            self.object = s3_object
            self.key = self.object.key
    
        def put(self, data):
            put_data = data
            if isinstance(data, str):
                try:
                    put_data = open(data, "rb")
                except IOError:
                    logger.exception("Expected file name or binary data, got '%s'.", data)
                    raise
    
            try:
                self.object.put(Body=put_data)
                self.object.wait_until_exists()
                logger.info(
                    "Put object '%s' to bucket '%s'.",
                    self.object.key,
                    self.object.bucket_name,
                )
            except ClientError:
                logger.exception(
                    "Couldn't put object '%s' to bucket '%s'.",
                    self.object.key,
                    self.object.bucket_name,
                )
                raise
            finally:
                if getattr(put_data, "close", None):
                    put_data.close()
                    
            self.object.Acl().put(ACL='public-read')

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("goingcrazy")
    
    obj_wrapper = ObjectWrapper(bucket.Object(f"Questions{clip_num}.mp3"))
    obj_wrapper.put(os.path.join(fake_audios_path, f"Questions{clip_num}.mp3"))
    
    obj_wrapper = ObjectWrapper(bucket.Object(f'Questions{clip_num}.mp4'))
    obj_wrapper.put(os.path.join(concat_path, f"Questions{clip_num}.mp4"))
    
    obj_wrapper = ObjectWrapper(bucket.Object(f"Summary{clip_num}.mp3"))
    obj_wrapper.put(os.path.join(fake_audios_path, f"Summary{clip_num}.mp3"))
    
    obj_wrapper = ObjectWrapper(bucket.Object(f'Summary{clip_num}.mp4'))
    obj_wrapper.put(os.path.join(concat_path, f"Summary{clip_num}.mp4"))
    
    print("upload_files cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def synclabs_api(clip_num):
    """
    對Sync labs傳送API請求，1秒的對嘴影片要花費大約10秒
        
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
    """
    
    print("Start executing synclabs_api")
    function_start_time = timeit.default_timer()

    payload = {
        "audioUrl": f"https://goingcrazy.s3.ap-northeast-1.amazonaws.com/Questions{clip_num}.mp3",
        "model": "sync-1.6.0",
        "synergize": True,
        "videoUrl": f"https://goingcrazy.s3.ap-northeast-1.amazonaws.com/Questions{clip_num}.mp4"
    }
    
    headers = {
        "x-api-key": "ae2d3598-ff37-49ad-9aec-4640fedacf3e",
        "Content-Type": "application/json"
    }
    
    # 送出請求
    url = "https://api.synclabs.so/lipsync"
    response = requests.request("POST", url, json=payload, headers=headers)
    
    ID = json.loads(response.text).get('id')
    url = f"https://api.synclabs.so/lipsync/{ID}"
    headers = {"x-api-key": "ae2d3598-ff37-49ad-9aec-4640fedacf3e"}
    done = False
    
    # 等待請求完成
    while not done:
        response = requests.request("GET", url, headers=headers)
        
        if json.loads(response.text).get('status') == 'COMPLETED':
            url_link = json.loads(response.text)['url']
            urllib.request.urlretrieve(url_link, os.path.join(synclabs_path, f'Questions{clip_num}.mp4'))
            done = True
            
    payload = {
        "audioUrl": f"https://goingcrazy.s3.ap-northeast-1.amazonaws.com/Summary{clip_num}.mp3",
        "model": "sync-1.6.0",
        "synergize": True,
        "videoUrl": f"https://goingcrazy.s3.ap-northeast-1.amazonaws.com/Summary{clip_num}.mp4"
    }
    
    headers = {
        "x-api-key": "ae2d3598-ff37-49ad-9aec-4640fedacf3e",
        "Content-Type": "application/json"
    }
    
    # 送出請求
    url = "https://api.synclabs.so/lipsync"
    response = requests.request("POST", url, json=payload, headers=headers)

    ID = json.loads(response.text).get('id')
    url = f"https://api.synclabs.so/lipsync/{ID}"
    headers = {"x-api-key": "ae2d3598-ff37-49ad-9aec-4640fedacf3e"}
    done = False
    
    # 等待請求完成
    while not done:
        response = requests.request("GET", url, headers=headers)
        
        if json.loads(response.text).get('status') == 'COMPLETED':
            url_link = json.loads(response.text)['url']
            urllib.request.urlretrieve(url_link, os.path.join(synclabs_path, f'Summary{clip_num}.mp4'))
            done = True
            
    print("synclabs_api cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def delete_files_from_s3(clip_num):
    """
    刪除上傳Amazon S3的檔案
        
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
    """
    
    print("Start executing delete_files")
    function_start_time = timeit.default_timer()
    logger = logging.getLogger(__name__)
    
    class ObjectWrapper:
        def __init__(self, s3_object):
            self.object = s3_object
            self.key = self.object.key
            
        def delete(self):
            try:
                self.object.delete()
                self.object.wait_until_not_exists()
                logger.info(
                    "Deleted object '%s' from bucket '%s'.",
                    self.object.key,
                    self.object.bucket_name,
                )
            except ClientError:
                logger.exception(
                    "Couldn't delete object '%s' from bucket '%s'.",
                    self.object.key,
                    self.object.bucket_name,
                )
                raise

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("goingcrazy")
        
    obj_wrapper = ObjectWrapper(bucket.Object(f"Questions{clip_num}.mp3"))
    obj_wrapper.delete()
    
    obj_wrapper = ObjectWrapper(bucket.Object(f'Questions{clip_num}.mp4'))
    obj_wrapper.delete()
    
    obj_wrapper = ObjectWrapper(bucket.Object(f"Summary{clip_num}.mp3"))
    obj_wrapper.delete()
    
    obj_wrapper = ObjectWrapper(bucket.Object(f'Summary{clip_num}.mp4'))
    obj_wrapper.delete()
    
    print("delete_files cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def add_subtitles(clip_num):
    """
    添加字幕
        
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
        
    檔案需求
    ---------------------
    ./data/sync_labs/Summary{clip_num}.mp4
    ./data/sync_labs/Questions{clip_num}.mp4
    ./data/sync_labs/Summary{clip_num}.txt
    ./data/sync_labs/Questions{clip_num}.txt
    """
    
    print("Start executing add_subtitles")
    function_start_time = timeit.default_timer()
    generator = lambda txt: TextClip(txt, font='Microsoft-JhengHei-Bold-&-Microsoft-JhengHei-UI-Bold', fontsize=80, color='white', stroke_color="black", stroke_width=2)
        
    if not os.path.exists(os.path.join(synclabs_path, f"Summary{clip_num}.mp4")) or not os.path.exists(os.path.join(synclabs_path, f"Questions{clip_num}.mp4")):
        print(f'{clip_num}的對嘴影片不存在')
        return
    
    texts = pd.read_csv(os.path.join(fake_audios_path, f"Summary{clip_num}.txt"), sep=' ', header=None)
    subs = [((start_time, end_time), text) for index, (text, start_time, end_time) in texts.iterrows()]
    
    subtitles = SubtitlesClip(subs, generator)
    video = VideoFileClip(os.path.join(synclabs_path, f"Summary{clip_num}.mp4"))
    result = CompositeVideoClip([video,  subtitles.set_pos(('center', 950))])
    result.write_videofile(os.path.join(subtitle_added_path, f"Summary{clip_num}.mp4"), logger=None)
    
    texts = pd.read_csv(os.path.join(fake_audios_path, f"Questions{clip_num}.txt"), sep=' ', header=None)
    subs = [((start_time, end_time), text) for index, (text, start_time, end_time) in texts.iterrows()]
    
    subtitles = SubtitlesClip(subs, generator)
    video = VideoFileClip(os.path.join(synclabs_path, f"Questions{clip_num}.mp4"))
    result = CompositeVideoClip([video,  subtitles.set_pos(('center', 950))])
    result.write_videofile(os.path.join(subtitle_added_path, f"Questions{clip_num}.mp4"), logger=None)
        
    print("add_subtitles cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def concat_results(clip_num):
    """
    串接切割片段、Questions、Summary
        
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
        
    檔案需求
    ---------------------
    ./data/sync_labs/Summary{clip_num}.mp4
    ./data/sync_labs/Questions{clip_num}.mp4
    ./data/clips/{clip_num}.mp4
    """
    
    print("Start executing concat_results")
    function_start_time = timeit.default_timer()
    
    if not os.path.exists(os.path.join(clips_path, f"{clip_num}.mp4")) or not os.path.exists(os.path.join(subtitle_added_path, f"Summary{clip_num}.mp4")) or not os.path.exists(os.path.join(subtitle_added_path, f"Questions{clip_num}.mp4")):
        print(f'{clip_num}的對嘴影片、分割片段不存在')
        return
        
    video = VideoFileClip(os.path.join(clips_path, f"{clip_num}.mp4"))
    summary = VideoFileClip(os.path.join(subtitle_added_path, f"Summary{clip_num}.mp4"))
    questions = VideoFileClip(os.path.join(subtitle_added_path, f"Questions{clip_num}.mp4"))
    result = concatenate_videoclips([video, summary, questions])
    result.write_videofile(os.path.join(results_path, f"{clip_num}.mp4"), logger=None)
            
    print("concat_results cost %d seconds\n" % (timeit.default_timer() - function_start_time))
        
def test():
    """
    測試用函數
    """
    pass
    

# 切割後片段的路徑
clips_path = r"./data/clips"

# 問題摘要文字檔的路徑
fake_audios_path = r"./data/ans"

# 句子分割檔的路徑
subclips_path = r"./data/subclips"

# 模型參數的儲存路徑
checkpoints_path = r"./checkpoints"

# 模型參數的讀取路徑
checkpoint_name = "checkpoint_0.34146_0.39837_simple.pt"

# 最終成果的路徑
results_path = r"./data/results"

# frames路徑
frames_path = r"./data/source_frames"

# 裁切路徑
crop_path = r'./data/crop'

# 拼接影片的片段來源路徑
concat_source_path = r'./data/concat_source'

# 拼接影片結果的路徑
concat_path = r'./data/concat'

# Synclabs影片的儲存路徑
synclabs_path = r'./data/synclabs'

subtitle_added_path = r'./data/subtitle_added'

# 資料的路徑
data_path = r"./data"

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_labels = 5
    
    # 下方為生成影片主要步驟
    # create_video_for_summary_and_questions(1)
    # upload_files_to_s3(1)
    # synclabs_api(1)
    # delete_files_from_s3(1)
    # add_subtitles(1)
    # concat_results(1)

