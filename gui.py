import streamlit as st
import requests

st.title("Next Word Prediction System")

seed_text = st.text_input("Enter seed text:")
next_words = st.number_input("Number of words to generate:", min_value=1, max_value=50, value=5)
temperature = st.slider("Temperature (creativity):", 0.1, 1.5, 0.7)

if st.button("Generate"):
    payload = {
        "seed_text": seed_text,
        "next_words": next_words,
        "temperature": temperature
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        st.write("Prediction:", response.json()["prediction"])
    else:
        st.write("Error:", response.text)
