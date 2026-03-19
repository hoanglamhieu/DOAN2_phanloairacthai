♻️ Waste Classification System using CNN
📌 Giới thiệu

Dự án xây dựng hệ thống phân loại rác thải sinh hoạt sử dụng mạng nơ-ron tích chập (CNN - Convolutional Neural Network).

Hệ thống cho phép người dùng:

Upload ảnh rác thải

Mô hình AI sẽ tự động nhận diện và phân loại

👉 Ứng dụng giúp hỗ trợ phân loại rác, nâng cao ý thức bảo vệ môi trường.

🎯 Mục tiêu

Xây dựng mô hình AI phân loại rác

Nhận diện nhiều loại rác khác nhau

Xây dựng web demo trực quan, dễ sử dụng

🧠 Công nghệ sử dụng

Python

TensorFlow / Keras

OpenCV / PIL

Flask (Web)

HTML, CSS

📂 Cấu trúc thư mục
phanloairacthaisinhhoat/
│
├── dataset_new/              # Dataset huấn luyện
│   ├── battery/
│   ├── biological/
│   ├── brown-glass/
│   ├── cardboard/
│   ├── clothes/
│   ├── green-glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   ├── trash/
│   └── white-glass/
│
├── model/                   # Lưu model sau khi train
│
├── static/                  # CSS, ảnh
│
├── waste_classification_web/ # Code web Flask
│
├── train_model.py           # File train CNN
├── class_names.json         # Danh sách label
├── requirements.txt         # Thư viện
├── README.md
└── .gitignore
🧩 Dataset

Dataset gồm nhiều loại rác:

Battery (pin)

Biological (rác hữu cơ)

Glass (kính: xanh, trắng, nâu)

Cardboard (bìa carton)

Clothes (quần áo)

Metal (kim loại)

Paper (giấy)

Plastic (nhựa)

Shoes (giày dép)

Trash (rác khác)

👉 Dữ liệu được tổ chức theo từng thư mục → phù hợp để train CNN.

🏗️ Kiến trúc hệ thống
User → Upload ảnh → Tiền xử lý → CNN Model → Predict → Hiển thị kết quả
🧠 Mô hình CNN

Mô hình bao gồm:

Convolution Layer: trích xuất đặc trưng ảnh

MaxPooling Layer: giảm kích thước

Flatten Layer

Dense Layer

Output Layer (Softmax)

👉 Sử dụng TensorFlow/Keras để xây dựng và huấn luyện.

⚙️ Cài đặt
1. Clone project
git clone https://github.com/your-username/phanloairacthaisinhhoat.git
cd phanloairacthaisinhhoat
2. Cài thư viện
pip install -r requirements.txt
🚀 Train model
python train_model.py

Model sẽ được lưu trong thư mục model/

🌐 Chạy web
cd waste_classification_web
python app.py

Truy cập:

http://127.0.0.1:5000
🔍 Dự đoán

Upload ảnh từ giao diện web

Hệ thống trả về:

Loại rác

Độ tin cậy (confidence)

📊 Kết quả

Mô hình đạt độ chính xác khoảng: 80% - 95% (tuỳ dataset)

Nhận diện tốt các loại rác phổ biến

🔧 Hướng phát triển

Tăng dữ liệu thực tế

Data Augmentation

Nâng cấp model:

ResNet

MobileNet

EfficientNet

Deploy lên web/app

Nhận diện realtime bằng camera
