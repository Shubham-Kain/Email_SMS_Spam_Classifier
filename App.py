from zipfile import sizeFileHeader

import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sklearn



ps =PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text

tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))
model =pickle.load(open("spam_classifier.pkl","rb"))

st.title("Email and SMS Spam Classifier")

input_sms = st.text_area("Enter the message",height=280,)
if st.button('Predict'):
 transform_sms = transform_text(input_sms)

 vector_input = tfidf.transform([transform_sms])

 result = model.predict(vector_input)[0]

 if result==1:
    st.header("Spam")
 else:
     st.header("Not Spam")
