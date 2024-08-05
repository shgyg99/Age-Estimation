import streamlit as st
from PIL import Image
import torch
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import Webcam, picture_upload, AgeEstimationModel
from inputs import models, main_path
from transforms import test_transform
AgeEstimationModel()

st.write("""
## Age Estimination App
""")
st.write('---')

# st.sidebar.header('choose one of these options')

# pic = st.sidebar.button("upload a picture", type="secondary")
# camera = st.sidebar.button('take picture using webcam', type="secondary")

col1, col2 = st.columns(2)
with col1:
    up = st.button('upload a picture', key=1)
    if up==1:
        up = st.file_uploader('upload', type=['png', 'jpg'])
        if up is not None:
            
            bar = st.progress(0)
            time.sleep(1)
            bar.progress(30)
            time.sleep(1)
            bar.progress(70)
            time.sleep(1)
            bar.progress(100)
            with open(up.name,'wb') as f:
                try:
                    f.write(up.read())
                    a = st.image(up)
                    file_path = os.path.join(main_path, str(up.name))
                    b = st.warning('just a sec')
                    out = picture_upload(models['resnet50'], file_path, test_transform)
                    st.balloons()
                    time.sleep(1)
                    a.image(out)
                    f.close()
                    os.remove(file_path)
                    
                except:
                    f.close()
                    os.remove(os.path.join(main_path, up.name))
                    b.warning('change your file name to a valid one like <me.png>')
    
with col2:
    pic = st.button('take a picture', key=2)
    if pic==2:
        pic = st.camera_input('take a photo')
        if pic is not None:
            pass
    
