import kagglehub
import os
import shutil

# Tải phiên bản mới nhất của Garbage Classification
path = kagglehub.dataset_download("mostafaabla/garbage-classification")

print("Path to dataset files:", path)

# Di chuyển dữ liệu về thư mục dự án của bạn cho dễ quản lý
destination = "dataset_new"
if not os.path.exists(destination):
    shutil.copytree(path, destination)
    print(f"✅ Đã copy dữ liệu về thư mục: {destination}")