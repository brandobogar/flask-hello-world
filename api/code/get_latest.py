import os
import glob
import re

def get_latest_uploads():
    fd_path = r"dataset/uploads"
    file_type = "/*jpg"
    files = glob.glob(fd_path+file_type)
    citra = max(files, key=os.path.getctime)
    latest_uploads = re.sub(r'\\', '/', citra)
    print('Citra:',latest_uploads)
    return latest_uploads

def get_latest_result():
    fd_path = r"dataset/result"
    file_type = "/*jpg"
    files = glob.glob(fd_path+file_type)
    citra = max(files, key=os.path.getctime)
    latest_result = re.sub(r'\\', '/', citra)
    print('Citra:',latest_result)
    return latest_result