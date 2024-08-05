import streamlit as st
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from functions import Webcam, picture_upload, AgeEstimationModel
from inputs import models
from transforms import test_transform
AgeEstimationModel()

st.write("""
## Age Estimination App
""")
st.write('---')

st.sidebar.header('choose one of these options')

pic = st.sidebar.button("upload a picture", type="secondary")
camera = st.sidebar.button('take picture using webcam', type="secondary")

# st.toast('hi')
# st.balloons()
# st.snow()
up = st.file_uploader('upload', type=['png', 'jpg'])
if up.read() is not None:
    
    bar = st.progress(0)
    time.sleep(1)
    bar.progress(30)
    time.sleep(1)
    bar.progress(70)
    time.sleep(1)
    bar.progress(100)
    st.image(up)
    file = up.read()
    out = picture_upload(models['resnet50'], file, test_transform)
    st.image(out)

pic = st.camera_input('take a picture')
