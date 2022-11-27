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
#nltk.download('all')

import re

# Importing module
@st.cache(ttl=24*60*60)

df = pd.read_csv('LTrain.csv')
df.drop(columns='Unnamed: 0', inplace=True)
@st.cache(ttl=24*60*60)

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


def processing(data):
  lt = WordNetLemmatizer()
  corpus = []
  for item in data:
    new_item = re.sub('[^a-zA-Z]',' ',str(item))
    new_item = new_item.lower()
    new_item = new_item.split()
    new_item = [lt.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(str(x) for x in new_item))
  return corpus
corpus = processing(df['Reviews'])

@st.cache(ttl=24*60*60)

X =corpus
y = df.Sentiments.values
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.20, random_state = 0)

tf_x_train =  tfidf.fit_transform(X_train).toarray()
from sklearn.linear_model import SGDClassifier
model= SGDClassifier()
model.fit(tf_x_train,y_train)

@st.cache(ttl=24*60*60)
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
        process_input = processing([[user_input]])
        vector_input = tfidf.transform(process_input)
        predictions = model.predict(vector_input)
        #return predictions
        #predictions = load_pred()
        if predictions==1:
            st.success("Positive Reviews👍")
 
        elif predictions==0:
            st.success("Negative Reviews 👎")
        else:
            st.error("Wrong Input")
                  

        
        


if __name__=='__main__':
    main()
