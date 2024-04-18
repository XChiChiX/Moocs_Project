import os
import zipfile

def zip_mp4_files():
    zip_file_name = 'result.zip'
    results_path = "./data/results"
    # 創建壓縮檔案
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        # 遍歷指定資料夾中的所有檔案
        for root, dirs, files in os.walk(results_path):
            for file in files:
                # 檢查檔案是否是 mp4 檔案
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    # 將 mp4 檔案添加到壓縮檔案中
                    zipf.write(file_path, os.path.relpath(file_path, results_path))

# 指定資料夾路徑和要建立的壓縮檔案名稱
# folder_path = '/path/to/mp4/files'

# 壓縮 mp4 檔案
# zip_mp4_files(zip_file_name)
