from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# =========================
# Tạo thư mục upload
# =========================
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# Load model
# =========================
model = load_model("./model/waste_cnn.h5")

# ⚠️ PHẢI GIỐNG 100% LÚC TRAIN
class_names = ['metal', 'organic', 'paper', 'plastic']

# Map sang tiếng Việt
label_map = {
    'metal': 'Kim loại',
    'organic': 'Rác hữu cơ',
    'paper': 'Giấy',
    'plastic': 'Nhựa'
}

# =========================
# Hàm dự đoán + softmax
# =========================
def predict_waste(img_path):
    img = image.load_img(
        img_path,
        target_size=(224, 224),
        color_mode='rgb'
    )

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 👉 Softmax
    predictions = model.predict(img_array)[0]

    class_index = np.argmax(predictions)
    confidence = predictions[class_index] * 100

    label_en = class_names[class_index]
    label_vi = label_map[label_en]

    # 👉 Chuẩn bị softmax để đưa ra web
    softmax_dict = {}
    for i, prob in enumerate(predictions):
        softmax_dict[label_map[class_names[i]]] = round(float(prob * 100), 2)

    return label_vi, round(confidence, 2), softmax_dict


# =========================
# Route chính
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    img_path = None
    softmax = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            result, confidence, softmax = predict_waste(img_path)

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        img_path=img_path,
        softmax=softmax
    )


# =========================
# Run app
# =========================
if __name__ == '__main__':
    app.run(debug=True)
