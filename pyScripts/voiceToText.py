# import tkinter as tk
# from tkinter import filedialog
import whisper

# root = tk.Tk()
# root.title("轉換的時刻到了")          #視窗標題
# root.geometry('700x500')              #視窗尺寸 

# file_path = ""

# def update():
#     pathLabel.configure(text=file_path)

#     print("Update path success...")

# def selectFile():
#     global file_path
#     file_path = filedialog.askopenfilename()   # 選擇檔案後回傳檔案路徑與名稱
#     print(file_path)                          # 印出路徑
#     update()

def generateTextFile(file_path):
    print("Converting...")

    with open("./data/clips/textAndTime_1.txt", "w", encoding="UTF-8") as file:
        file.write("文字+開始時間。\n\n")
    # with open("textOnly.txt", "w", encoding="UTF-8") as file:
    #     file.write("純文字。\n\n")
    
    result = voiceToText(file_path)

    for seg in result["segments"]:
        # print(seg["text"], round(seg["start"], 1), round(seg["end"], 1))
        buffer = seg["text"] + " " + (str)(round(seg["start"], 1))
        with open("./data/clips/textAndTime_1.txt", "a", encoding="UTF-8") as output_file:
            output_file.write(buffer)
            # output_file.write('\n')
    # with open("textOnly.txt", "a", encoding="UTF-8") as output_file:
    #     output_file.write(result["text"])
    #     output_file.write('\n')
    
    print("Text writing completed...")
    # finish()


def voiceToText(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)

    # print(result["text"])

    return result

# def finish():
#     finishLabel = tk.Label(root, text="轉換完畢!!", font=('Arial',30))  # 放入標籤，使用 textvariable=text
#     finishLabel.place(x=200, y=200)


# # Button 設定 command 參數，點擊按鈕時執行函式
# selectBtn = tk.Button(root,
#                 text='選擇檔案',
#                 font=('Arial',10,'bold'),
#                 command=selectFile
#             )

# confirmBtn = tk.Button(root,
#                 text='啟動',
#                 font=('Arial',10,'bold'),
#                 command=start
#             )

# pathLabel = tk.Label(root, text=file_path, font=('Arial', 10)) #字體與大小

# # 位置
# selectBtn.pack(anchor=tk.NW)
# pathLabel.place(x=100, y=0)
# confirmBtn.pack(anchor=tk.NE)

# root.mainloop()