import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk import word_tokenize
from nltk.tokenize import word_tokenize
#libraries for lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('averaged_perceptron_tagger')
#nltk.download('all')

import re
import dill
import pickle
import joblib

file1 = open('processing_function.pkl0', 'rb')
file2 = open('sentiment_model.pkl0', 'rb')
file3 =  open('fitted_tfidf1.pkl0', 'rb')
processor = dill.load(file1)
predictor = joblib.load('new_model')
fitted = dill.load(file3)

# Importing module

from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
tfidf = tf_idf(ngram_range=(1,4),
          min_df=3, max_df=0.9, use_idf=1,
          smooth_idf=1, sublinear_tf=1, binary=bool)



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
        #def load_pred():
        #data = pd.DataFrame({'Reviews':[user_input]})
        process_input = processor([[user_input]])
        #vector_input = tfidf.transform(process_input)
        predictions = predictor.predict(process_input)
        #return predictions
        #predictions = load_pred()
        if predictions==1:
            st.success("Positive Reviews👍")
 
        else:
            st.success("Negative Reviews 👎")

        
        


if __name__=='__main__':
    main()
