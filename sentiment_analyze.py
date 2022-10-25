import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
#from nltk import word_tokenize
from nltk.tokenize import word_tokenize
#libraries for lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('averaged_perceptron_tagger')
nltk.download('all')

import re
import dill
import pickle


file1 = open('processing_function.pkl', 'rb')
file2 = open('sentiment_model.pkl', 'rb')
file3 =  open('fitted_tfidf1.pkl', 'rb')
processor = dill.load(file1)
predictor = pickle.load(file2)
fitted = dill.load(file3)

# Importing module

from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
tfidf = tf_idf(ngram_range=(1,5),
          min_df=2, max_df=0.9, strip_accents='unicode', use_idf=1,
          smooth_idf=1, sublinear_tf=1,analyzer='word',norm='l2')



# Importing module
st.set_page_config(
    page_title="Sentiment Analysis For Online Reviews",
    page_icon="🖥️",
    layout="wide",
)

X_train = np.load('X_train.npy')
X_train =  tfidf.fit_transform(X_train).toarray()


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
        vector_input = tfidf.transform(process_input)
        predictions = predictor.predict(vector_input)
        #return predictions
        #predictions = load_pred()
        if predictions==1:
            st.success("Positive Reviews👍")
 
        else:
            st.success("Negative Reviews 👎")

        
        


if __name__=='__main__':
    main()
