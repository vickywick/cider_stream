from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from abels import label
from os import listdir
from os.path import isfile, join
from PIL import Image,ImageOps
import io

model = load_model('apple-224.h5')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def import_predict(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    x=np.amax(prediction)
    return np.argmax(prediction),x 

def run():
    image_leafy= Image.open('leafy.jpg')
    st.sidebar.info('This web app is created for Plant Diesease Detection')
    st.title("Plant Disease  Prediction App")
    st.sidebar.image(image_leafy)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    onlyfiles = [f for f in listdir("img/") if isfile(join("img/", f))]
    if st.sidebar.checkbox("Trial Run",False,key='1'):
        file_upload = st.sidebar.selectbox("Choose an image for Disease-Prediction", onlyfiles)
        if file_upload is not None:
            path="img/"+file_upload
            image = Image.open(path)
            st.sidebar.image(image,caption="Uploaded Image",use_column_width=True)
            if st.sidebar.button('Predict'): 
                st.sidebar.write("Classifying...")
                lab,x=import_predict(image)
                output=label[lab]
                st.success('The prediction is {} {}'.format(output,(x*100)))
    img_upload = st.file_uploader("Choose an image for Pediction", type=["jpg","png"])

    if img_upload is not None:
        path=img_upload
        image = Image.open(path)
        st.image(image,caption="Uploaded Image",use_column_width=True)
        if st.button('Predict',key=2): 
            y="classifying...."
            st.write(y)
            lab , x =import_predict(image)
            y="classified :)"
            st.write(y)
            output=label[lab]
            st.success('The prediction is {} {}'.format(output,(x*100)))
    
if __name__ == '__main__':
    run()