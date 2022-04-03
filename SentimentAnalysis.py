import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import warnings
warnings.filterwarnings(action='ignore')


model2 = tf.keras.models.load_model("model2")

st.set_page_config(page_title='Analisis Sentimen Pada Review Hotel')


st.title('Analisis Sentimen Pada Review Hotel')
st.header('Masukkan Review Hotel')

contoh_review= st.text_area('Untuk hasil yang lebih baik, review lebih dari satu elemen')

if contoh_review:
    prediksi = model2.predict([contoh_review])
    if prediksi.squeeze()>0.5:
        st.write('Sentimen Positif')
        st.image('Thumbs Up.png')
    else:
        st.write('Sentimen Negatif')
        st.image('Thumbs Down.png')
else:
    pass
