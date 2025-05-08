import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained model
model = load_model("face_mask.model.h5")
IMG_SIZE = 100

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to predict mask or no mask
def predict_face(face_img):
    try:
        face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face)
        label = "with Mask" if np.argmax(prediction) == 0 else "without Mask"
        return label
    except:
        return "Error"

# Webcam face mask detection
def webcam_detection():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x, y, box_w, box_h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    face_img = frame[y:y+box_h, x:x+box_w]
                    label = predict_face(face_img)
                    color = (0, 255, 0) if label == "with Mask" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display the webcam feed on Streamlit
            stframe.image(frame, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()

# Image upload for prediction
def predict_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x, y, box_w, box_h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face_img = img_rgb[y:y+box_h, x:x+box_w]
                label = predict_face(face_img)
                color = (0, 255, 0) if label == "with Mask" else (0, 0, 255)
                cv2.rectangle(img_rgb, (x, y), (x + box_w, y + box_h), color, 2)
                cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    st.image(img_rgb, channels="RGB")

# Streamlit App
def main():
    st.title("Face Mask Detection using Streamlit and CNN")

    st.write("Choose a mode of detection:")

    # Option 1: Webcam Detection
    if st.button("Start Webcam Detection"):
        st.write("Starting webcam detection...")
        webcam_detection()

    # Option 2: Upload an Image for Prediction
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.write("Image uploaded. Making prediction...")
        predict_uploaded_image(uploaded_file)

# Run the Streamlit app
if __name__ == "__main__":
    main()
