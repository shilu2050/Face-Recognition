import numpy as np
import cv2
import pickle
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
face_path = os.path.join(BASE_DIR, "cascades", "haarcascade_frontalface_alt2.xml")
eye_path = os.path.join(BASE_DIR, "cascades", "haarcascade_eye.xml")
smile_path = os.path.join(BASE_DIR, "cascades", "haarcascade_smile.xml")
trainer_path = os.path.join(BASE_DIR, "trainer.yml")
labels_path = os.path.join(BASE_DIR, "labels.pickle")

# Load cascades
face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
smile_cascade = cv2.CascadeClassifier(smile_path)

# Safety check
if face_cascade.empty():
    print("❌ Face cascade not loaded")
    exit()

# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists(trainer_path):
    print("❌ trainer.yml not found. Run faces-train.py first")
    exit()

recognizer.read(trainer_path)

# Load labels
with open(labels_path, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Predict face
        id_, conf = recognizer.predict(roi_gray)

        if 4 <= conf <= 85:
            name = labels.get(id_, "Unknown")
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()