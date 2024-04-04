import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.title("轉換的時刻到了")          #視窗標題
root.geometry('700x500')              #視窗尺寸 

def update():
    pathLabel.configure(text=file_path)

    print("Update path success...")

def selectFile():
    global file_path
    file_path = filedialog.askopenfilename()   # 選擇檔案後回傳檔案路徑與名稱
    print(file_path)                          # 印出路徑
    update()

def start():
    # voiceToText()
    finish()

def finish():
    finishLabel = tk.Label(root, text="轉換完畢!!", font=('Arial',30))  # 放入標籤，使用 textvariable=text
    finishLabel.place(x=200, y=200)

file_path = ""

selectBtn = tk.Button(root,
                text='選擇檔案',
                font=('Arial',10,'bold'),
                command=selectFile
            )

confirmBtn = tk.Button(root,
                text='啟動',
                font=('Arial',10,'bold'),
                command=start
            )

pathLabel = tk.Label(root, text=file_path, font=('Arial', 10)) #字體與大小

# 位置
selectBtn.pack(anchor=tk.NW)
pathLabel.place(x=100, y=0)
confirmBtn.pack(anchor=tk.NE)

root.mainloop()