import streamlit as st
# import numpy as np
from flask import Flask, render_template, redirect, url_for, request
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import numpy as np
from PIL import Image

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

model = load_model('..\Model\Emotion_Detector.h5')

classifier = cv2.CascadeClassifier('..\Haar Cascade\haarcascade_frontalface_default.xml')

def make_prediction(image):
    img = Image.open(image)
    img = np.array(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img, scaleFactor=1.1)
    for (x, y, w, h) in faces:
        # n+=1
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face/255
        emotion = model.predict(np.array([face]))
        emotion = labels[np.argmax(emotion)]
        print(emotion)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = cv2.putText(img, emotion, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return Image.fromarray(img)


file = st.file_uploader(label='Select Image', type=['png', 'jpg', 'jpeg'])

if st.button(label='Submit'):
    st.image(make_prediction(file), width=640)

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Khan Faisal</p><br>
<a href='https://khanfaisal.netlify.app'>  Portfolio</a>
<a href="https://github.com/khanfaisal79960">  Github</a>
<a href="https://medium.com@khanfaisal79960">  Medium</a>
<a href="https://www.linkedin.com/in/khanfaisal79960">  Linkedin</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
