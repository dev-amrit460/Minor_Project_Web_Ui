from keras.models import load_model
from asyncio.windows_events import NULL
from flask import Flask, render_template, Response
import cv2
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import string
import os
import glob
from PIL import Image
from time import time
from tensorflow.python import keras
import threading
from keras import Input, layers
from keras import optimizers
#from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2 import adam
from keras_preprocessing import sequence
from keras_preprocessing import image
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout

from keras.layers.merging import add
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet_v2 import ResNet50V2
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.efficientnet_v2 import preprocess_input
from keras.models import Model
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.utils.np_utils import to_categorical

model_efficientnet_v2 = EfficientNetV2L(
    include_top=True,
    weights="imagenet",
    pooling='avg',
    classifier_activation="softmax",
)
caption="No caption, Kindly refresh!"
model_EfficientNetV2L = Model(model_efficientnet_v2.input, model_efficientnet_v2.layers[-2].output)

caption_model_v2 = load_model('EfficientNetV2L_lstm_model.h5')
max_length = 74
vocab = open('vocab.txt', 'r').read().strip().split('\n')

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(480, 480))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image_byme):
    image = preprocess(image_byme)
    fea_vec = model_EfficientNetV2L.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model_v2.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = caption_model_v2.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

app = Flask(__name__)
camera = NULL
flag = True


def generate_frames(camera):
    global flag
    while flag:

        # Camera frames
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def camera_off():
#     flag=False
#     # camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    # flag=True
    global camera
    camera = cv2.VideoCapture(0)
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/videoff')
def videoff():
    global flag
    flag = False
    global camera
    camera.release()
    return render_template('index.html')


@app.route('/videon')
def videon():
    global flag
    flag = True
    return render_template('index.html')


@app.route('/capture')

def capture():
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    result, img = cam.read()
    res = cv2.resize(img, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
    if result:
        x = image.img_to_array(res)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fea_vec = model_EfficientNetV2L.predict(x)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        encoded_img1 = fea_vec.reshape(1, 1280)
        caption= beam_search_predictions(encoded_img1)
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(caption)
        engine.runAndWait()
        return render_template('index.html',string_variable=caption)
    return render_template('index.html',string_variable="Capture Failed")



if __name__ == "__main__":
    app.run(debug=True)
