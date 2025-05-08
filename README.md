# face-mask-detection-deep-learning
This project is a deep learning-based face mask detection system that determines whether a person is wearing a face mask or not, using real-time computer vision techniques. It supports both webcam input and uploaded images for prediction. The system is designed to be lightweight, efficient, and suitable for real-world deployment.

**Key Features:**
Real-time face mask detection: Detects face mask usage using webcam or uploaded image.

High Accuracy: Achieved over 98% accuracy on the validation dataset.

Interactive Interface: Built with Streamlit for easy deployment and interaction.

Real-time Face Detection: Uses OpenCV and MediaPipe for real-time face detection and facial landmark detection.

**Tech Stack:**
Python: Programming language for model development and application.

TensorFlow & Keras: Used for building and training the custom Convolutional Neural Network (CNN).

OpenCV: For video processing and face detection.

MediaPipe: Used for detecting facial landmarks.

Streamlit: Web framework to create the interactive app for users to upload images or use the webcam.

NumPy, pandas: For data manipulation and preprocessing.

**Model Performance:**
The CNN model was trained on a dataset with two categories: with_mask and without_mask.

Training Accuracy: 98%

Validation Accuracy: 98%

Model Type: Convolutional Neural Network (CNN)
