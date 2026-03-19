from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model = load_model(r"C:\phanloairacthaisinhhoat\model\waste_mobilenet_v2.h5")


with open("class_names.json", "r") as f:
    class_names = json.load(f)


label_map = {
    'battery': 'Pin độc hại',
    'biological': 'Rác hữu cơ/Sinh học',
    'brown-glass': 'Thủy tinh nâu',
    'cardboard': 'Bìa carton',
    'clothes': 'Quần áo cũ',      
    'green-glass': 'Thủy tinh xanh',
    'metal': 'Kim loại',
    'paper': 'Giấy',
    'plastic': 'Nhựa',
    'shoes': 'Giày dép cũ',
    'trash': 'Rác hỗn hợp',
    'white-glass': 'Thủy tinh trắng'
}

def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

  
    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index] * 100

   
    str_index = str(class_index)
    label_en = class_names.get(str_index, "Unknown")
    
   
    label_vi = label_map.get(label_en, label_en)

   
    softmax_dict = {}
    for i, prob in enumerate(predictions):
        name_en = class_names.get(str(i), f"Lớp {i}")
        name_vi = label_map.get(name_en, name_en)
        softmax_dict[name_vi] = round(float(prob * 100), 2)

    return label_vi, round(float(confidence), 2), softmax_dict

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    img_path = None
    softmax = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = file.filename
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)
            
           
            img_display_path = 'uploads/' + filename 

           
            result, confidence, softmax = predict_waste(full_path)
            img_path = img_display_path

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        img_path=img_path,
        softmax=softmax
    )



if __name__ == '__main__':
    # Lấy cổng từ môi trường của Render, nếu không có thì mặc định là 5000
    port = int(os.environ.get('PORT', 5000))
    # Chạy ứng dụng với host 0.0.0.0
    app.run(host='0.0.0.0', port=port)