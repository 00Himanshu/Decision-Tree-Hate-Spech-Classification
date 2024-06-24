import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model_filename = "hate_speech_classifier.pkl"
vectorizer_filename = "vectorizer.pkl"

clf = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

# Function to classify new text
def classify_text(model, vectorizer, text):
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    # Predict the class
    prediction = model.predict(text_vec)
    # Map the prediction to labels
    labels = ['Race', 'Religion', 'Sexuality', 'Age']
    result = {labels[i]: prediction[0][i] for i in range(len(labels))}
    return result

# Streamlit app
st.title("Hate Speech Classification")

user_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if user_input:
        classification_result = classify_text(clf, vectorizer, user_input)
        st.write("Classification Result:", classification_result)
    else:
        st.write("Please enter some text to classify.")