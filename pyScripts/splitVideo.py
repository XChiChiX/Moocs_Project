from moviepy.editor import VideoFileClip
import os

def split_video(input_video):
    clip = VideoFileClip(input_video)
    total = clip.duration
    video_name = input_video.split('.')[0]
    
    start_times = get_start_times()
    print(start_times)
    end_times = [start_times[1], start_times[2], total]  # 结束时间列表，单位为秒
    
    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        sub_clip = clip.subclip(start_time, end_time)
        output_file = f"./data/clips/{video_name}_part_{i+1}.mp4"
        sub_clip.write_videofile(output_file, codec="libx264")

    clip.close()

    print("split_video completed...")

def get_start_times():
    # 打开文本文件并读取内容
    with open('./data/clips/timestamp.txt', 'r') as file:
        data = file.read()

    # 使用 split() 方法按空白字符分割字符串，得到字符串列表
    numbers_str = data.split()

    # 将字符串列表中的元素转换为整数，得到整数列表
    numbers_list = [float(num) for num in numbers_str]

    return numbers_list


# 使用示例
# input_video = "moocs1.mp4"  # 输入视频文件名

# split_video(input_video)

