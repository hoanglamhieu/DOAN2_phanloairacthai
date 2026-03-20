# ♻️ Waste Classification System using CNN

## 📌 Giới thiệu

Trong bối cảnh ô nhiễm môi trường ngày càng gia tăng, việc phân loại rác thải đóng vai trò rất quan trọng trong xử lý và tái chế.

Dự án này xây dựng một hệ thống **phân loại rác thải tự động** sử dụng **Deep Learning (CNN)**, cho phép người dùng tải ảnh lên và nhận kết quả phân loại ngay lập tức.

---

## 🎯 Mục tiêu

* Phân loại rác thải thành các nhóm:

  * 🟢 Rác hữu cơ
  * 🔵 Rác tái chế
  * ⚫ Rác vô cơ
* Ứng dụng mô hình CNN để nhận diện hình ảnh
* Xây dựng web app đơn giản để demo hệ thống

---

## 🧠 Công nghệ sử dụng

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Flask (Web Framework)
* HTML/CSS

---

## 📂 Cấu trúc thư mục

```
waste-classification/
│
├── dataset/              # Dữ liệu huấn luyện
├── model/                # Mô hình đã train (.h5)
├── static/               # Ảnh upload
├── templates/            # Giao diện HTML
├── app.py                # Chương trình chính (Flask)
├── train.py              # File huấn luyện model
└── requirements.txt      # Danh sách thư viện
```

---

## ⚙️ Hướng dẫn cài đặt

### 1. Clone project

```
git clone https://github.com/hoanglamhieu/DOAN2_phanloairacthai
cd waste-classification
```

---

### 2. Tạo môi trường ảo (khuyến nghị)

#### Windows:

```
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Cài đặt thư viện

```
pip install -r requirements.txt
```

---

## 🚀 Cách chạy chương trình

### ▶️ Chạy web app

```
python app.py
```

👉 Sau đó mở trình duyệt tại:

```
http://127.0.0.1:5000
```

---

### 🧪 Huấn luyện lại model (nếu cần)

```
python train.py
```

---

## 🔄 Quy trình hoạt động

```
Upload ảnh → Xử lý ảnh → CNN → Dự đoán → Hiển thị kết quả
```

---

## 📸 Demo hệ thống

* Người dùng upload ảnh rác
* Hệ thống phân tích bằng CNN
* Trả về kết quả phân loại

---

## ⚠️ Lỗi thường gặp

### ❌ Không import được TensorFlow

```
pip install tensorflow==2.10
```

---

### ❌ Thiếu thư viện

```
pip install <tên-thư-viện>
```

---

### ❌ Không load được model

* Kiểm tra file `.h5`
* Kiểm tra đường dẫn model

---

## 🔮 Hướng phát triển

* Tăng dữ liệu huấn luyện
* Áp dụng Data Augmentation
* Sử dụng mô hình nâng cao:

  * ResNet
  * MobileNet
  * EfficientNet
* Deploy lên web/server

---

## 👨‍💻 Tác giả

* Sinh viên: Hoàng Lâm Hiếu 
* Đề tài: **Phân loại rác thải bằng AI**

---

## ⭐ Đóng góp

Nếu bạn thấy dự án hữu ích, hãy ⭐ repo để ủng hộ!

---
