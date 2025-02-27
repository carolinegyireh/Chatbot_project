import numpy as np
import random
import json
import pickle
import torch
import gradio as gr
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Load necessary files
model = load_model('../data/models/med-botmodel.keras')
words = pickle.load(open('../data/words.pkl', 'rb'))
classes = pickle.load(open('../data/classes.pkl', 'rb'))
with open("../data/med_quad.json", "r") as json_file:
    dict_ = json.load(json_file)

# Load the BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

lemmatizer = WordNetLemmatizer()

# Function to get BERT embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to predict the intent of a user's input
# Converts the sentence into BERT embeddings and predicts the intent based on the model's output probabilities.
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    pred_intent= [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return pred_intent


# Function to get the response based on the intent predicted
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']  # Get the predicted intent
        for intent in intents_json['intents']:
            if intent['qtype'] == tag:  # Check if intent matches qtype in med_quad.son
                return intent['answer']  # Return the corresponding answer
    return "Sorry, I don't understand."  # Default response if no match

# Function that Gradio will use to provide chatbot responses
def chatbot_response(message):
    intents = predict_class(message)
    response = get_response(intents, dict_)
    return response

# The Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="Healthcare Chatbot",
    description="Hi! I'm your medical chatbot. What medical question can I help with today?",
    theme="huggingface"
)

# Launch the Gradio interface
iface.launch()
