import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("😊 Emotion Detector AI")
st.write("Detect emotions using webcam")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()

camera = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            prediction = emotion_model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame_placeholder.image(frame, channels="BGR")

camera.release()