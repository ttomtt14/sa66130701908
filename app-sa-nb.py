
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import streamlit as st

# Load the model from the file
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("Simple Sentiment Predictor")

# User input for text
text_input = st.text_input("Enter text:")

# Predict sentiment
if st.button("Predict"):
    prediction = loaded_model.predict([text_input])
    st.write("Prediction:", prediction[0])
