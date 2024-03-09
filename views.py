from flask import Flask, render_template, redirect, url_for, request
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import numpy as np

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

model = load_model('./Model/Emotion_Detector.h5')

classifier = cv2.CascadeClassifier('./Haar Cascade/haarcascade_frontalface_default.xml')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        file_name = image.filename
        image.save(f'./static/uploads/{file_name}')

        img = cv2.imread(f'./static/uploads/{file_name}')
        # img = img.resize((224, 224))
        # img = img_to_array(img)
        # img = img/255
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
            img = cv2.putText(img, emotion, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        img = cv2.resize(img, (224, 224))
        cv2.imwrite(f'./static/img/{file_name}', img)
                
        img_path = f'img/{file_name}'

        return render_template('index.html', img_path=img_path)

    return render_template('index.html')