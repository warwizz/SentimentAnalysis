#!/usr/bin/env python
# coding: utf-8

# ## Import library

# In[67]:


import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
pd.set_option("display.max_colwidth", 1000)
pd.set_option('display.max_rows', None)


# In[2]:


data =  pd.read_csv('review_hotel.csv',encoding='latin1')


# In[3]:


data


# ## PreProcessing

# ### Menghilangkan spasi, tanda baca, dan membuat huruf menjadi huruf kecil

# In[4]:


punctuations = re.sub(r"[!<_>#:)\.]", "", string.punctuation)

def punct2wspace(text):
    return re.sub(r"[{}]+".format(punctuations), " ", text)

def normalize_wspace(text):
    return re.sub(r"\s+", " ", text)

def casefolding(text):
    return text.lower()

def preprocess_text(text):
    text = punct2wspace(text)
    text = normalize_wspace(text)
    text = casefolding(text)
    return text

preprocess_text("Wc jorok.. Kasur tidak dibersihkan,, handuk tidak diganti,")


# In[5]:


data["cleaned_review"] = data["review_text"].apply(preprocess_text)
data.head()


# In[6]:


datax = data.drop(['review_id','review_text'],axis=1)


# In[7]:


datax


# ## EDA

# In[8]:


yes = datax[datax.category==1]
no = datax[datax.category==0]


# In[9]:


pd.DataFrame(yes['cleaned_review'].str.findall("(:\S+)").explode().value_counts())


# In[10]:


count_yes = pd.Series(' '.join(yes["cleaned_review"]).split()).value_counts()[:100]
count_yes


# In[11]:


col = ['Index','Kata','Jumlah']
count_yes = pd.DataFrame(count_yes).reset_index()
count_yes.columns
count_yes = count_yes.rename({'index':'Kata',0:'Jumlah'},axis=1)


# In[12]:


count_yes


# In[13]:


count_no = pd.Series(' '.join(no["cleaned_review"]).split()).value_counts()[:100]
count_no


# In[14]:


col = ['Index','Kata','Jumlah']
count_no = pd.DataFrame(count_no).reset_index()
count_no.columns
count_no = count_no.rename({'index':'Kata',0:'Jumlah'},axis=1)


# In[15]:


count_no


# In[16]:


plt.figure(figsize=[15,10])
plt.title('Jumlah Kata Pada Sentimen Negatif', fontsize=20)
sns.barplot(x='Kata',y='Jumlah',data=count_no[:20])
plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=45, fontsize=15)
plt.xlabel('Kata',fontsize=15)
plt.ylabel('Jumlah',fontsize=15)
plt.show()
plt.figure(figsize=[15,10])
plt.title('Jumlah Kata Pada Sentimen Positif', fontsize=20)
sns.barplot(x='Kata',y='Jumlah',data=count_yes[:20])
plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=45, fontsize=15)
plt.xlabel('Kata',fontsize=15)
plt.ylabel('Jumlah',fontsize=15)
plt.show()


# ->Hal yang sering muncul/dibahas pada sentimen positif adalah **```bersih, nyaman, kamar, bagus, ramah, harga, pelayanan, baik, tempat, dekat```**
# ->Hal yang sering muncul/dibahas pada sentimen negaitf adalah **```kamar, mandi, air, airy, kotor, ac, hotel, bau```**

# In[17]:


bagus = " ".join(review for review in datax[datax["category"]==1].cleaned_review)
jelek = " ".join(review for review in datax[datax["category"]==0].cleaned_review)


# In[18]:


wordcloud = WordCloud(background_color="white",max_words=100).generate(bagus)

# Display the generated image:
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[19]:


wordcloud = WordCloud(background_color="white",max_words=100).generate(jelek)

# Display the generated image:
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Split Data

# In[20]:


Review = datax.cleaned_review
Cat = datax.category


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(Review, Cat, random_state=42)


# In[22]:


BATCH_SIZE = 32


# In[23]:


Review


# ## Tensorflow Pipeline

# In[24]:


trainset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
testset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

trainset = trainset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
testset = testset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Hal di atas dilakukan supaya data bisa digunakan oleh model di tensorflow

# In[25]:


for feat, tar in trainset.take(1):
    print(feat[:3])
    print(tar[:3])


# ## Cek Tokenisasi Kata

# In[26]:


max_features = 10000
embedding_dim = 16


# In[27]:


encoder = keras.layers.TextVectorization(max_tokens= max_features)
encoder.adapt(trainset.map(lambda feat, tar:feat))


# In[28]:


encoder(feat)[:3]


# In[29]:


encoder.get_vocabulary()


# # Membangun Model

# ## Model 1

# In[30]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=1e-2, patience=10, verbose=1, restore_best_weights=True)
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=1e-1, patience=5, verbose=1, min_delta=1e-2)


# In[31]:


model = keras.Sequential()
model.add(encoder)
model.add(keras.layers.Embedding(
    input_dim=len(encoder.get_vocabulary()), 
    output_dim=embedding_dim, 
    mask_zero=True)
)
model.add(keras.layers.LSTM(16, return_sequences=True))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics=["accuracy"])


# In[32]:


model.fit(trainset, validation_data=testset, batch_size=BATCH_SIZE, 
    epochs=100, callbacks=[early_stop, reduce_lr_on_plateau])


# In[40]:


y_pred = model.predict(X_test)


# In[56]:


mat = confusion_matrix(y_test,np.round(y_pred))
plot_confusion_matrix(conf_mat=mat)


# In[63]:


print(classification_report(y_test,np.round(y_pred),labels = [1,0]))


# ## Model 2

# In[33]:


model2 = keras.Sequential()
model2.add(encoder)
model2.add(keras.layers.Embedding(
    input_dim=len(encoder.get_vocabulary()), 
    output_dim=embedding_dim, 
    mask_zero=True)
)
model2.add(keras.layers.LSTM(16, return_sequences=True))
model2.add(keras.layers.LSTM(16))
model2.add(keras.layers.Dropout(0.5))
model2.add(keras.layers.Dense(8))
model2.add(keras.layers.Dense(1, activation="sigmoid"))

model2.compile(loss="binary_crossentropy", 
              optimizer="rmsprop",
              metrics=["accuracy"])


# In[34]:


model2.fit(trainset, validation_data=testset, batch_size=BATCH_SIZE, 
    epochs=100, callbacks=[early_stop, reduce_lr_on_plateau])


# In[58]:


y_pred2 = model2.predict(X_test)


# In[59]:


mat = confusion_matrix(y_test,np.round(y_pred2))
plot_confusion_matrix(conf_mat=mat)


# In[64]:


print(classification_report(y_test,np.round(y_pred2),labels = [1,0]))


# ## Cek Model Terhadap Contoh Review Baru

# In[53]:


contoh_review = ['Tempat parkir luas, kamarnya nyaman dan bersih, pelayannya ramah']
prediksi = model2.predict(contoh_review) # Probabilitas
prediksi.squeeze()
if prediksi.squeeze()>0.5:
    print(prediksi.squeeze())
    print('Sentimen Positif')
else:
    print(prediksi.squeeze())
    print('Sentimen Negatif')


# In[71]:


st.set_page_config(page_title='Sentiment Analysis on Hotel Review')
    
st.sidebar.header('Please Input Hotel Review')
contoh_review = st.sidebar.text_input("For better result, please review more than one element")


# In[ ]:





# In[ ]:





# In[ ]:




