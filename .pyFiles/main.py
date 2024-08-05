import streamlit as st
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
uploaded = None
st.write("""
## Age Estimination App
""")
st.write('---')

st.sidebar.header('choose one of these options')

pic = st.sidebar.button("upload a picture")
camera = st.sidebar.button('take picture using webcam')

# st.toast('hi')
# st.balloons()
# st.snow()
up = st.file_uploader('upload', type=['png', 'jpg'])
if up is not None:
    bar = st.progress(0)
    time.sleep(1)
    bar.progress(30)
    time.sleep(1)
    bar.progress(70)
    time.sleep(1)
    bar.progress(100)
    st.image(up)
    

elif camera:
    picture = st.camera_input('take a picture')

