import io
import os
import base64
import numpy as np
import cv2
from PIL import Image
import requests
import streamlit as st

# Function to detect Braille using Roboflow direct HTTP API
def detect_braille(image_bytes, conf_threshold=0.25):
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
        return None, []

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

def main():
    st.title("Braille Detection with Roboflow")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        with st.spinner("Detecting Braille..."):
            try:
                img_str, text_rows = detect_braille(image_bytes, conf_threshold)
                if img_str is None:
                    st.warning("No Braille detected.")
                else:
                    st.image(base64.b64decode(img_str), caption="Detected Braille", use_column_width=True)
                    st.subheader("Detected Text Rows")
                    for row in text_rows:
                        st.write(row)
            except Exception as e:
                st.error(f"Error during detection: {e}")

if __name__ == "__main__":
    main()
