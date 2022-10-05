import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image

st.set_page_config(
    page_title="Sentiment Analysis For Online Reviews",
    page_icon="🖥️",
    layout="wide",
)

def main():
    st.title('Online Review Analyser')
    image = Image.open('https://github.com/Heartz00/Online-review-analyzer/blob/792bd573ac7d67ef0ce9291be64122a6ba8d902b/analyze_pics.jpg')
    n_image = image.resize((500,300))
    st.image(n_image)
    st.write('Welcome to the review analyser, you can check the sentiment of products reviews here in few seconds')
    form = st.form(key='sentiment-form')
    user_input = form.text_area('Enter your text')
    submit = form.form_submit_button('Submit')

    analyzer= SentimentIntensityAnalyzer()
    if submit:
        st.info('Result')
        sentiment_dict = analyzer.polarity_scores(user_input)
        st.write("The review is  ", sentiment_dict['neg']*100, "% Negative")
        st.write("The review is  ", sentiment_dict['neu']*100, "% Neutral")
        st.write("The review is  ", sentiment_dict['pos']*100, "% Positive")
        st.write("Overall Review Analysis :", end = " ")
        if sentiment_dict['compound'] >= 0.05 :
            st.success("Positive 👍")
 
        elif sentiment_dict['compound'] <= - 0.05 :
            st.success("Negative 👎")
 
        else :
            st.success("Neutral😐")
        
        


if __name__=='__main__':
    main()


