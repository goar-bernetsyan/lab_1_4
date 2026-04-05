from flask import Flask, render_template, request, jsonify, url_for
import os
import uuid
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загрузите модель YOLOv5
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')

# Названия классов (по умолчанию из модели)
class_names = model.names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Нет файла'}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Загрузка изображения для модели
    img = cv2.imread(filepath)
    results = model(img)
    df = results.pandas().xyxy[0]

    # Рисуем рамки на изображении
    for _, row in df.iterrows():
        print(row)
        # xmin, ymin, xmax, ymax, conf, cls = row
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        conf = row['confidence']
        cls = row['class']

        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        class_id = int(cls)
        color = (0, 255, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_names[class_id]} {conf:.2f}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    result_filename = f"{uuid.uuid4().hex}_result.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)

    return jsonify({'result_image_url': url_for('static', filename='uploads/' + result_filename)})

if __name__ == '__main__':
    app.run(debug=True)