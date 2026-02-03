**# 🤟 AI Sign Language Recognition System

An AI-based real-time Sign Language Recognition system that detects and classifies hand gestures using **OpenCV**, **MediaPipe**, and **TensorFlow**.

This project recognizes sign language alphabets (A, B, C) from live webcam input and displays predictions with confidence scores.

---

## 📌 Features
- Real-time hand detection using MediaPipe
- Gesture classification using a CNN model
- Live webcam prediction with bounding box & landmarks
- Trained deep learning model (`.h5`)
- Modular step-by-step pipeline (data collection → training → prediction)

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **NumPy**
- **Scikit-learn**

---

## 📂 Project Structure
```text
**AI-Sign-Language-Recognition/
│
├── data/                     # Collected gesture images
│   ├── A/
│   ├── B/
│   └── C/
│
├── step1_test.py              # Camera & environment test
├── step3_hand_detection.py    # Hand landmark detection
├── step4_collect_data.py      # Dataset collection
├── step5_train_model.py       # Model training
├── step6_live_prediction.py   # Real-time prediction
│
├── sign_model.h5              # Trained model
├── requirements.txt
├── .gitignore
└── README.md**
**
