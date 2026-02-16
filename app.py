from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model and tokenizer
model = load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

seq_length = 73  # same as training

# FastAPI setup
app = FastAPI()

# Input schema
class PredictRequest(BaseModel):
    seed_text: str
    next_words: int = 5
    temperature: float = 0.7

# Temperature sampling function
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# API endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    text_for_test = req.seed_text
    for _ in range(req.next_words):
        token_text = tokenizer.texts_to_sequences([text_for_test])[0]
        padded_token = pad_sequences([token_text], maxlen=seq_length, padding='pre')
        predicted_probs = model.predict(padded_token, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted_probs, req.temperature)
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                text_for_test += " " + word
                break
    return {"prediction": text_for_test}
