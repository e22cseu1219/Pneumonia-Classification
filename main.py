import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image
import numpy as np
from util import classify, set_background

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  
        super().__init__(*args, **kwargs)


get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D


set_background('./bgs/bg5.png')


st.title('Pneumonia classification')


st.header('Please upload a chest X-ray image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model(r'C:\Users\shash\aiproject\pneumonia-classification-web-app-python-streamlit\model\pneumonia_classifier.h5')
model.save('model_compatible.h5')  

with open(r'C:\Users\shash\aiproject\pneumonia-classification-web-app-python-streamlit\model\labels.txt') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
