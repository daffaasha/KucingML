import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import load_model

def convertImage(mbek_url):
    img_height = 180
    img_width = 180

    mbek_path = tf.keras.utils.get_file(origin=mbek_url)

    img = tf.keras.utils.load_img(
    mbek_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    return img_array

def loadModel():
    return load_model('model/artmodel.h5')

def predict(model):
    predictions = model.predict(processedImg)
    score = tf.nn.softmax(predictions[0])
    classNames = ['Albrecht Dürer', 'Alfred Sisley', 'Francisco Goya', 'Pablo Picasso']
    print(classNames)

    st.subheader(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(classNames[np.argmax(score)], 100 * np.max(score)))


st.title("Who's Art Is This")

mbekImage = st.text_input('Input Art Link', placeholder='Give Me Art')

if mbekImage:
    try:
        processedImg = convertImage(mbekImage)
    except Exception as e:
        print(e)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(mbekImage, width=300)
    processBtn = st.button("Process")

with col3:
    st.write(' ')

if processBtn:
    predict(loadModel())