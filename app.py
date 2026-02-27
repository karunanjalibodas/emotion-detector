import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load models
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("😊 Emotion Detector AI")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            prediction = emotion_model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img

webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)
