from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip
import pandas as pd
import glob
import os
import timeit
from ultralytics import YOLO
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import whisper
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, BertTokenizer, BertModel
import numpy as np
import librosa
from pydub import AudioSegment 
import shutil
import random
import requests
import json
import urllib.request
from transparent_background import Remover
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from sklearn.cluster import KMeans
import boto3
from botocore.exceptions import ClientError
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

def same_seeds(seed):
    """
    固定種子
    """
    
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
# 聲音轉文字
def voice_to_text(path):
    function_start_time = timeit.default_timer()
    model = whisper.load_model("medium")
    file_name = os.path.basename(path)
    
    print("Start executing voice_to_text")
    result = model.transcribe(path)
    with open(os.path.join(os.path.dirname(path),  file_name.split('.')[0] + ".txt"), "w", encoding="utf-8") as f:
        for segment in result['segments']:
            f.write(segment['text'].replace(' ', '') + ' ' + (str)(round(segment['start'], 2)) + ' ' + (str)(round(segment['end'], 2)) + '\n')
                
    print("voice_to_text cost %ds\n" % (timeit.default_timer() - function_start_time))

def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """
    切割影片
    """
    
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)
    
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%t1,
           "-i", filename,
           "-t", "%0.2f"%(t2-t1),
           targetname]

    subprocess_call(cmd, logger=None)

def extract_subclips():
    """
    以句子為單位分割影片
    """
    
    print('Start executing extract_subclips')
    function_start_time = timeit.default_timer()
    clips_list = glob.glob(os.path.join(clips_path, "*.mp4")) + glob.glob(os.path.join(clips_path, "*.mkv"))
    subclips_info = pd.DataFrame()
    model = YOLO('yolov8x.pt')
    
    if os.path.exists(os.path.join(subclips_path, "unclassified")) == False:
        os.mkdir(os.path.join(subclips_path, "unclassified"))
    
    for clip in clips_list:
        print("開始處理%s的影片分割" % clip)
        clip_name = os.path.basename(clip).split('.')[0]
        text = pd.read_csv(os.path.join(clips_path, os.path.basename(clip).split('.')[0] + ".txt"), sep=' ', header=None)
        for index, (sentence, start_time, end_time) in text.iterrows():
            
            # 捨棄小於1秒的片段
            if end_time - start_time < 1:
                continue
            
            ffmpeg_extract_subclip(clip, start_time, end_time, targetname=os.path.join(subclips_path, "unclassified","%s_%d.mp4" % (clip_name ,index)))
            
            # 每5幀抽樣來確認整個片段中都有人
            cap = cv2.VideoCapture(os.path.join(subclips_path, "unclassified","%s_%d.mp4" % (clip_name ,index)))
            one_person_in_clip = True
            
            while cap.isOpened():
                ret, frame = cap.read()

                if ret is False:
                    break
                
                result = model(frame, classes=0, verbose=False)[0]
                           
                if len(result.boxes) != 1:
                    one_person_in_clip = False
                    break
                
                for i in range(4):
                    cap.grab()

            cap.release()
            
            if one_person_in_clip:
                subclips_info = pd.concat((subclips_info, pd.DataFrame([["%s_%d.mp4" % (clip_name ,index), sentence]])), ignore_index=True)
            
    subclips_info.to_csv(os.path.join(subclips_path, "subclips_info.csv"), encoding='utf-8', index=False)    
    print("extract_subclips cost %ds\n" % (timeit.default_timer() - function_start_time))
      
def mp4_to_mp3():
    """
    將每個句子分割檔轉成mp3
    """
    function_start_time = timeit.default_timer()
    print("Start executing mp4_to_mp3")
    
    # 未分類片段的列表
    subclips_list = glob.glob(os.path.join(subclips_path, "unclassified", "*.*"))
    
    # 確認儲存音檔資料夾已建立
    if os.path.exists(os.path.join(subclips_path, "unclassified_audio")) == False:
        os.mkdir(os.path.join(subclips_path, "unclassified_audio"))
    
    # 轉成mp3
    for subclip in subclips_list:
        video = VideoFileClip(subclip)
        video.audio.write_audiofile(os.path.join(subclips_path, "unclassified_audio", os.path.basename(subclip).split(".")[0] + ".mp3"), logger=None)
        
    print("mp4_to_mp3 cost %ds\n" % (timeit.default_timer() - function_start_time))
    
def compute_subclips_finger_movement():
    """
    記錄每一筆資料的平均食指移動量
    """
    
    print('Start executing compute_subclips_finger_movement')
    function_start_time = timeit.default_timer()
    
    # MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=r'./checkpoints/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    subclips_info = pd.read_csv(os.path.join(subclips_path, "subclips_info.csv"))
    
    for index, (clip_name, sentence) in subclips_info.iterrows():
        print("\r開始測量%s的食指移動量" % clip_name)
        cap = cv2.VideoCapture(os.path.join(subclips_path, "unclassified", clip_name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = total_frame / fps
        movement = 0
        last_left_index_finger = None
        last_right_index_finger = None
        
        while cap.isOpened():
            ret, frame = cap.read()

            if ret is False:
                break
            
            # 用MediaPipe函式庫讀取當前的frame
            cv2.imwrite(r'./data/temp/frame.jpg', frame)
            image = mp.Image.create_from_file(r'./data/temp/frame.jpg')
            
            # 偵測手部節點
            detection_result = detector.detect(image)
            
            # 儲存食指座標
            left_index_finger = None
            right_index_finger = None
            for i, handedness in enumerate(detection_result.handedness):
                if handedness[0].category_name == "Left":
                    left_index_finger = detection_result.hand_landmarks[i][8]
                elif handedness[0].category_name == "Right":
                    right_index_finger = detection_result.hand_landmarks[i][8]
            
            # 計算並累積食指移動量
            if last_left_index_finger != None and left_index_finger != None:
                movement += math.sqrt((last_left_index_finger.x - left_index_finger.x) ** 2 + (last_left_index_finger.y - left_index_finger.y) ** 2 + (last_left_index_finger.z - left_index_finger.z) ** 2)

            if last_right_index_finger != None and right_index_finger != None:
                movement += math.sqrt((last_right_index_finger.x - right_index_finger.x) ** 2 + (last_right_index_finger.y - right_index_finger.y) ** 2 + (last_right_index_finger.z - right_index_finger.z) ** 2)
            
            # 更新上一次的食指座標
            if left_index_finger != None:
                last_left_index_finger = left_index_finger
                
            if right_index_finger != None:
                last_right_index_finger = right_index_finger

            # 跳過4幀
            for i in range(4):
                cap.grab()
                
        cap.release()
        
        # 紀錄食指每秒平均移動量
        subclips_info.loc[index, "2"] = movement / length
    
    subclips_info.to_csv(os.path.join(subclips_path, "subclips_info.csv"), encoding='utf-8', index=False)
    print("compute_subclips_finger_movement cost %ds\n" % (timeit.default_timer() - function_start_time))

def label_subclips(num_labels, method):
    """
    標註每一筆訓練資料
    """
    
    print("Start executing label_subclips")
    function_start_time = timeit.default_timer()
    subclips_info = pd.read_csv(os.path.join(subclips_path, "subclips_info.csv"))
    
    # 將移動量為0的值設為平均移動量
    avg = subclips_info.iloc[:, 2].replace(0, np.NaN).mean()
    subclips_info.iloc[:, 2] = subclips_info.iloc[:, 2].replace(0, avg)
    
    # 移動量由小到大排序
    subclips_info = subclips_info.sort_values("2")
    
    if method == "Equal":
        
        # 計算類別間的門檻值
        threshold = [subclips_info.iloc[round((subclips_info.shape[0] - 1) * (i + 1) / num_labels), 2] for i in range(num_labels - 1)]
        
        # 取得移動量對應的類別編號
        def get_class(movement, threshold):
            for index, t in enumerate(threshold):
                if movement < t:
                    return index
            return len(threshold)
        
        # 標註資料類別
        subclips_info["3"] = subclips_info.apply(lambda row: get_class((float)(row.iloc[2]), threshold), axis=1)
    
    # 使用K-means對資料做分群
    elif method == "Kmeans":
        
        # 分群
        kmeans = KMeans(n_clusters=num_labels, random_state=0)
        labels = kmeans.fit_predict(subclips_info[['2']])
        
        # 將標籤由小到大標註
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = np.array([center[0] for center in cluster_centers])
        sort_index = list(np.argsort(cluster_centers))
        labels = [sort_index.index(i) for i in labels]
        subclips_info['3'] = labels

    
    # 依資料編號排序
    subclips_info = subclips_info.sort_index()
    
    subclips_info.to_csv(os.path.join(subclips_path, "subclips_info.csv"), encoding='utf-8', index=False)
    
    # 將每個片段移至對應的類別資料夾
    for index, (clip_name, sentence, move_amount, class_num) in subclips_info.iterrows():
        
        # 確認類別資料夾存在
        if os.path.exists(os.path.join(subclips_path, (str)(class_num))) == False:
            os.makedirs(os.path.join(subclips_path, (str)(class_num)))
            
        shutil.copy(os.path.join(subclips_path, "unclassified", clip_name), os.path.join(subclips_path, (str)(class_num), clip_name))
        
    # 顯示標籤分布
    print("Label distribution:")
    print(subclips_info["3"].value_counts())
    
    print("label_subclips cost %ds\n" % (timeit.default_timer() - function_start_time))

class CustomDataset(Dataset):  
    def __init__(self, subclips_path, max_length):
        self.subclips_info = pd.read_csv(os.path.join(subclips_path, "subclips_info.csv"))
        
        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.texts = [self.tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") for text in self.subclips_info["1"]]
        
        # Wav2vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_names = [name.split('.')[0] + ".mp3" for name in self.subclips_info["0"]]
        self.features = [self.feature_extractor(librosa.load(os.path.join(subclips_path, "unclassified_audio", name), sr=16000)[0], return_tensors="pt", padding="max_length", max_length=60000, truncation=True, sampling_rate=16000) for name in self.audio_names]
        self.labels = [label for label in self.subclips_info["3"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.features[idx], self.labels[idx]

class Wav2Vec2BertClassifier(nn.Module):
    """
    Model for deciding the magnitude of movement according to text and audio.
    """
    
    def __init__(self, num_labels, dropout=0.2):
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
    
def train():
    """
    訓練模型
    """
    
    print("Start executing train")
    function_start_time = timeit.default_timer()
    
    # 計算每個類別的損失權重
    subclips_info = pd.read_csv(os.path.join(subclips_path, "subclips_info.csv"))
    y = subclips_info["3"].to_numpy()
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    model = Wav2Vec2BertClassifier(num_labels, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    dataset = CustomDataset(subclips_path, max_length=max_length)
    train_set, val_set, test_set = random_split(dataset, [0.7, 0.1, 0.2])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model = Wav2Vec2BertClassifier(num_labels, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    best_acc = -1.0
    
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        # Training
        model.train()
        for texts, audios, labels in train_loader: # tqdm(train_loader):
            input_wav2vec2 = audios["input_values"].squeeze(1).to(device)
            input_bert = texts["input_ids"].squeeze(1).to(device)
            mask_bert = texts["attention_mask"].squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad() 
            outputs = model(input_wav2vec2, input_bert, mask_bert) 
            
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            
            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for texts, audios, labels in val_loader: # tqdm(val_loader):
                input_wav2vec2 = audios["input_values"].squeeze(1).to(device)
                input_bert = texts["input_ids"].squeeze(1).to(device)
                mask_bert = texts["attention_mask"].squeeze(1).to(device)
                labels = labels.to(device)
                
                outputs = model(input_wav2vec2, input_bert, mask_bert)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) 
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()
                
        # print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} Loss: {val_loss/len(val_loader):3.5f}')

        if val_acc > best_acc:
            
            # 刪除上一個pt檔
            if os.path.exists(os.path.join(checkpoints_path, f'checkpoint_{best_acc/len(val_set):.5f}.pt')):
                os.remove(os.path.join(checkpoints_path, f'checkpoint_{best_acc/len(val_set):.5f}.pt'))
                
            best_acc = val_acc
            torch.save({"model_state_dict":model.state_dict()}, os.path.join(checkpoints_path, f"checkpoint_{best_acc/len(val_set):.5f}.pt"))
            # print(f'saving model with acc {best_acc/len(val_set):.5f}')

    # Test
    y_pred = []
    y_true = []
    test_acc = 0.0
    test_loss = 0.0
    model.load_state_dict(torch.load(os.path.join(checkpoints_path, f"checkpoint_{best_acc/len(val_set):.5f}.pt"))['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for texts, audios, labels in test_loader: # tqdm(test_loader):
            input_wav2vec2 = audios["input_values"].squeeze(1).to(device)
            input_bert = texts["input_ids"].squeeze(1).to(device)
            mask_bert = texts["attention_mask"].squeeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(input_wav2vec2, input_bert, mask_bert)
            
            loss = criterion(outputs, labels) 
            
            _, test_pred = torch.max(outputs, 1) 
            y_pred.extend(test_pred.view(-1).detach().cpu().numpy())       
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            test_acc += (test_pred.cpu() == labels.cpu()).sum().item()
            test_loss += loss.item()
            
    if show_cm:
        # cm = confusion_matrix(y_true, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.show()
        
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
    # 刪除上一個pt檔
    if os.path.exists(os.path.join(checkpoints_path, f'checkpoint_{best_acc/len(val_set):.5f}.pt')):
        os.remove(os.path.join(checkpoints_path, f'checkpoint_{best_acc/len(val_set):.5f}.pt'))
        
    torch.save({"model_state_dict":model.state_dict()}, os.path.join(checkpoints_path, f"checkpoint_{best_acc/len(val_set):.5f}_{test_acc/len(test_set):3.5f}.pt"))
    print(f'Test Acc: {test_acc/len(test_set):3.5f} Loss: {test_loss/len(test_loader):3.5f}')
            
    print("train cost %ds\n" % (timeit.default_timer() - function_start_time))
        
def video_to_images(path):
    """
    儲存影片中的每一偵圖片
    """
    
    print('Start executing video_to_images')
    function_start_time = timeit.default_timer()
    
    cap = cv2.VideoCapture(path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break
        
        cv2.imwrite(os.path.join(frames_path, f'frame_{count}.png'), frame)
        count += 1
        
    cap.release()
    print("video_to_images耗時 %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def process_animateanyone(path):
    """
    處理AnimateAnyone產生的影片
    """
    
    print('開始執行process_animateanyone')
    function_start_time = timeit.default_timer()
    video_name = os.path.basename(path)
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    remover = Remover(mode="base", jit=True, device="cuda:0")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break
        
        # 取出影片右方三分之一的部分
        frame = frame[:, int(width // 3 * 2):, :]
        frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]), cv2.INTER_LANCZOS4)
        
        # Padding至1920x1080
        frame = cv2.copyMakeBorder(frame, 1080 - frame.shape[0], 0, 1920 - frame.shape[1], 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 去背
        frame = Image.fromarray(frame)
        frame = np.array(remover.process(frame, type="white"))

        if writer == None:
            writer = cv2.VideoWriter(os.path.join(data_path, f'processed_{video_name}'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

        writer.write(frame)
    
    cap.release()
    writer.release()
    print("process_animateanyone cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
def crop_clips(mode='image'):
    """
    將人物置中並調整成512x512解析度
    """
    
    print('Start executing crop_clips')
    function_start_time = timeit.default_timer()
    
    model = YOLO(r'./checkpoints/yolov8x.pt')

    if mode == 'video':
        video_list = glob.glob(os.path.join(crop_path, '*.mp4'))
        
        for video_path in video_list:
            
            video_name = os.path.basename(video_path)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(os.path.join(crop_path, f'cropped_{video_name}'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))
            first = True
            
            while cap.isOpened():
                ret, frame = cap.read()
        
                if ret is False:
                    break
            
                result = model(frame, classes=0, retina_masks=True, verbose=False)[0]
                
                if first:
                    box = result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
                    expand_length = ((y2 - y1) - (x2 - x1)) // 2
                    left_pad_length = min(x1 - expand_length, 0) * (-1)
                    right_pad_length = max(x2 + expand_length, width) - width
                    first = False
                
                frame = frame[y1:y2, max(x1 - expand_length, 0):min(x2 + expand_length, width), :]
                frame = cv2.resize(cv2.copyMakeBorder(frame, 0, 0, left_pad_length, right_pad_length, cv2.BORDER_CONSTANT, value=[255, 255, 255]), (512, 512))
                writer.write(frame)
                
            cap.release()
            writer.release()
    elif mode == 'image':
        img_list = glob.glob(os.path.join(crop_path, '*.png'))
        
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            width = frame.shape[1]
            
            result = model(frame, classes=0, retina_masks=True, verbose=False)[0]
            
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
            expand_length = ((y2 - y1) - (x2 - x1)) // 2
            left_pad_length = min(x1 - expand_length, 0) * (-1)
            right_pad_length = max(x2 + expand_length, width) - width
            first = False
        
            frame = frame[y1:y2, max(x1 - expand_length, 0):min(x2 + expand_length, width), :]
            frame = cv2.copyMakeBorder(frame, 0, 0, left_pad_length, right_pad_length, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            cv2.imwrite(os.path.join(crop_path, f'cropped_{img_name}'), frame)
            
    
    print("crop_clips cost %d seconds\n" % (timeit.default_timer() - function_start_time))
    
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
    ./checkpoints/{checkpoint_name}
    """
    
    print("Start executing concat_clips")
    function_start_time = timeit.default_timer()
    model = Wav2Vec2BertClassifier(num_labels, dropout=dropout).to(device)
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
        audio_feature = feature_extractor(librosa.load(os.path.join(data_path, "temp.mp3"), sr=16000)[0], return_tensors="pt", padding="max_length", max_length=48000, truncation=True, sampling_rate=16000).to(device)
        
        # BERT
        text_feature = tokenizer(sentence, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt").to(device)
        
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
        text_feature = tokenizer(sentence, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt").to(device)
        
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

def upload_files(clip_num):
    """
    上傳假影片跟假音檔至AWS S3
    
    參數
    ---------------------
    clip_num: 整數、字串
        分割片段的編號
        
    檔案需求
    ---------------------
    ./data/ans/Summary{clip_num}.mp3
    ./data/ans/Summary{clip_num}.mp4
    ./data/ans/Questions{clip_num}.mp3
    ./data/ans/Questions{clip_num}.mp4
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
    
def delete_files(clip_num):
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
checkpoint_name = "checkpoint_0.36585_0.42683_simple.pt"

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
    num_epoch = 20
    learning_rate = 5e-6
    dropout = 0.2
    batch_size = 32
    max_length = 30
    label_smoothing = 0.0
    num_labels = 5
    method = "Kmeans" # [Kmeans, Equal]
    show_cm = True  
    
    same_seeds(1)
    # voice_to_text(r"C:\Users\Owner\Desktop\Moocs_Project\data\ans\Summary1.mp3")
    # extract_subclips()
    # mp4_to_mp3()
    # compute_subclips_finger_movement()
    # label_subclips(num_labels=num_labels, method=method)
    # for seed in range(10):
    #     same_seeds(seed + 1)
    #     print(f'Seed: {seed + 1}')
    #     train()
    # video_to_images()
    # crop_clips(mode='video')
    # process_animateanyone(r"C:\Users\Owner\Desktop\Moore-AnimateAnyone-master\output\source_0_768x768_3_1353.mp4")
    # test()
    
    # 下方為生成影片主要步驟
    # create_video_for_summary_and_questions(1)
    # upload_files(1)
    # synclabs_api(1)
    # delete_files(1)
    add_subtitles(1)
    # concat_results(1)

