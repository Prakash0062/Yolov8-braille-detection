import io
import os
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to detect Braille using Roboflow direct HTTP API
def detect_braille(image_path, conf_threshold=0.25):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Run inference using HTTP request
    response = requests.post(
        "https://detect.roboflow.com/braille-final-05-05/2",
        params={
            "api_key": "30oxLfVGmeadjEk3NGVB"
        },
        files={
            "file": image_bytes
        }
    )

    result = response.json()

    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    # Process detections
    detections = []
    for pred in result.get("predictions", []):
        if pred["confidence"] < conf_threshold:
            continue
        x_center, y_center = int(pred["x"]), int(pred["y"])
        width, height = int(pred["width"]), int(pred["height"])
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2

        detections.append({
            'box': [x1, y1, x2, y2],
            'confidence': round(pred["confidence"], 2),
            'label': pred["class"],
            'x_center': x_center,
            'y_center': y_center
        })

    if not detections:
        return "", []

    # Sort and group detections into rows
    detections.sort(key=lambda d: d['y_center'])
    row_thresh = 20
    rows = []
    current_row = []
    last_y = -100

    for det in detections:
        y = det['y_center']
        if abs(y - last_y) > row_thresh:
            if current_row:
                rows.append(current_row)
            current_row = [det]
            last_y = y
        else:
            current_row.append(det)
            last_y = (last_y + y) // 2

    if current_row:
        rows.append(current_row)

    detected_text_rows = []
    for row in rows:
        sorted_row = sorted(row, key=lambda d: d['x_center'])
        row_labels = [d['label'] for d in sorted_row]
        detected_text_rows.append(''.join(row_labels))

    # Draw boxes on image
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f'{label} {conf}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str, detected_text_rows

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    conf_threshold = float(request.form.get("confidence", 0.25))

    try:
        img_str, text_rows = detect_braille(image_path, conf_threshold)
        return jsonify({
            'detected_image': f'data:image/jpeg;base64,{img_str}',
            'detected_text_rows': text_rows
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
