import numpy as np
import json
import pickle
import torch
import streamlit as st
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random

# path to necessary files
MODEL_PATH = "data/models/med-botmodel.keras"
WORDS_PATH = "data/words.pkl"
CLASSES_PATH = "data/classes.pkl"
INTENTS_PATH = "data/med_quad.json"

# Load model and necessary data
model = load_model(MODEL_PATH)
words = pickle.load(open(WORDS_PATH, "rb"))
classes = pickle.load(open(CLASSES_PATH, "rb"))

# Load intents JSON file
with open(INTENTS_PATH, "r") as json_file:
    intents_data = json.load(json_file)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

lemmatizer = WordNetLemmatizer()

# Predefined responses for greetings and queries not in the knowledge base
greeting_responses = [
    "Hello! How can I assist you today?",
    "Hi there! What medical question do you have?",
    "Hey! Feel free to ask any health-related questions."
]

unknown_responses = [
    "I'm sorry, I don't have enough information on that. Could you rephrase?",
    "That's beyond my knowledge base. Maybe a healthcare professional can help."
]

# Function to generate BERT embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Predict intent function
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    predictions = model.predict(embedding)[0]

    ERROR_THRESHOLD = 0.25
    intent_results = [[i, p] for i, p in enumerate(predictions) if p > ERROR_THRESHOLD]
    intent_results.sort(key=lambda x: x[1], reverse=True)

    return [{'intent': classes[i], 'probability': str(p)} for i, p in intent_results]

# Get response function with improved fallback handling
def get_response(predicted_intents, message):
    """
    Returns an appropriate response based on the predicted intent.
    """
    # Check if the message is a greeting
    if message.lower() in ["hi", "hello", "hey", "good morning", "good evening"]:
        return random.choice(greeting_responses)

    # If no intent is confidently detected, return a fallback response
    if not predicted_intents:
        return random.choice(unknown_responses)

    # Extract intent tag
    intent_tag = predicted_intents[0]['intent']

    # Search for matching intent in dataset
    for intent in intents_data["intents"]:
        if intent["qtype"] == intent_tag:
            return intent["answer"]

    return random.choice(unknown_responses)

# Streamlit UI with custom styling
st.markdown("""
    <style>
    body {
        background-color: #f7e3e3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextInput input {
        border-radius: 20px;
        padding: 10px;
        font-size: 18px;
        width: 90%;
        border: 2px solid #f2a7a7;
    }
    .stButton>button {
        border-radius: 15px;
        background-color: #f8b0b0;
        color: white;
        padding: 12px 25px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff9b9b;
    }
    .stMarkdown {
        color: #333333;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-bubble {
        background-color: #1d72b8;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 12px;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #f3f3f3;
        color: #333333;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 12px;
        max-width: 80%;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Add chatbot title and introductory text
st.title("💖 Your Cute Healthcare Buddy 💖")
st.markdown("_Hi! I'm your healthcare chatbot. Ask me anything about medical conditions, and I'll do my best to help!_")

# Create a session state to store conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Text input for user message with placeholder
user_input = st.text_input("Type your question here...", "", key="user_input")

# Function to update and display chat history
if user_input:
    predicted_intents = predict_class(user_input)
    response = get_response(predicted_intents, user_input)
    st.session_state.history.append(("You: " + user_input, "Bot: " + response))

    # Clear the input box after submitting the question
    st.session_state.user_input = ""

# Display the chat history with styled chat bubbles
chat_container = st.container()
with chat_container:
    for user_message, bot_response in st.session_state.history:
        # Display user message as a chat bubble
        st.markdown(f'<div class="user-bubble">{user_message}</div>', unsafe_allow_html=True)
        # Display bot response as a chat bubble
        st.markdown(f'<div class="bot-bubble">{bot_response}</div>', unsafe_allow_html=True)

# Reset chat history button
if st.button("Reset Chat"):
    st.session_state.history = []
    st.session_state.user_input = ""  # Clear the user input field
