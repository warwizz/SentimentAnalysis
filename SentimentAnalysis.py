import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import warnings
warnings.filterwarnings(action='ignore')


model2 = tf.keras.models.load_model("model2")

st.set_page_config(page_title='Analisis Sentimen Pada Review Hotel')



st.title('Final Project Sentiment Analysis from Data Bangalore')
st.header('Project by Wisnu Waskitho Putra')
st.markdown("""
                    """)
st.markdown("""
                    """)
col1, col2, col3 = st.columns([2,2,2])
with col1:
    st.image("https://pbs.twimg.com/profile_images/1330698057143382016/FWcd26L5_400x400.jpg", width=200)

with col2:
    st.image("download.png", width=200)

with col3:
    st.image("https://scontent.fcgk37-1.fna.fbcdn.net/v/t1.6435-9/191062691_101547822145827_4402978100432027208_n.jpg?_nc_cat=103&ccb=1-5&_nc_sid=09cbfe&_nc_eui2=AeHcG-N75HVCNNYRJbuW0hjqK4TVCAIQRowrhNUIAhBGjCh_YjpsYlYpn0H7DrM7NZin32QVcDH8UhmflUkadTGM&_nc_ohc=Jw-ckmQdn5EAX-jnSjG&_nc_ht=scontent.fcgk37-1.fna&oh=00_AT_qXZOgO4gSOMog2aFarXjqOqmBP1bs0da-yytzJWEIQw&oe=627634CF", width=200)
    

st.header('Masukkan Review Hotel')

contoh_review= st.text_area('Untuk hasil yang lebih baik, review lebih dari satu elemen')

if contoh_review:
    prediksi = model2.predict([contoh_review])
    if prediksi.squeeze()>0.5:
        st.write('Sentimen Positif')
        st.image('Thumbs Up.png', width= 300)
    else:
        st.write('Sentimen Negatif')
        st.image('Thumbs Down.png', width= 300)
else:
    pass
