import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
#from nltk import word_tokenize
from nltk.tokenize import word_tokenize
#libraries for lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


import re
import dill
import pickle
import joblib

file1 = open('processing_.pkl', 'rb')
file3 =  open('fit_tfidf1.pkl', 'rb')
processor = dill.load(file1)
predictor = joblib.load('new-model')
fitted = dill.load(file3)
X_train = np.load('XX_traiN.npy')

# Importing module

from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
tfidf = tf_idf(ngram_range=(1,7),
          min_df=3, max_df=0.9, use_idf=1,
          smooth_idf=1, sublinear_tf=1, binary=bool)

tf_x_train =  tfidf.fit_transform(X_train.tolist()).toarray()

# Importing module
st.set_page_config(
    page_title="Sentiment Analysis For Online Reviews",
    page_icon="🖥️",
    layout="wide",
)



 
def main():
    st.title('Online Review Analyser')
    image = Image.open('emotions.png')
    n_image = image.resize((500,300))
    st.image(n_image)
    st.write('Welcome to the review analyser, you can check the sentiment of products reviews here in few seconds')
    form = st.form(key='sentiment-form')
    user_input = form.text_area('Enter your text')
    submit = form.form_submit_button('Submit')
    
    
    if submit:
        st.info('Result')
        process_input = processor([[user_input]])
        vector_input = tfidf.transform(process_input).toarray()
        predictions = predictor.predict(vector_input)
        #return predictions
        #predictions = load_pred()
        if predictions==1:
            st.success("Positive Reviews👍")
 
        elif predictions==0:
            st.success("Negative Reviews 👎")
        else:
            st.error("wrong input")




if __name__=='__main__':
    main()
