import numpy as np
import json
import pickle
import torch
import gradio as gr
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# path to necessary files
MODEL_PATH = "/content/med-botmodel.keras"
WORDS_PATH = "/content/words.pkl"
CLASSES_PATH = "/content/classes.pkl"
INTENTS_PATH = "/content/med_quad.json"

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

# Gradio chatbot function with conversation history
def chatbot_response(message, chat_history=[]):
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, message)
    chat_history.append(("You: " + message, "Bot: " + response))

    return chat_history, chat_history

# Gradio UI
with gr.Blocks(theme="huggingface") as interface:
    gr.Markdown("## Your Healthcare Buddy 🤖 ")
    gr.Markdown("_Hi! I'm your medical chatbot. What medical question can I help with today?_")

    chat_history = gr.State([])  # Stores conversation history

    with gr.Row():
        chatbot_display = gr.Chatbot()  # Chatbot UI
        reset_button = gr.Button("Reset Chat")

    user_input = gr.Textbox(label="Your Message", placeholder="Type your question here...")

    # update chat history when user sends a message
    user_input.submit(chatbot_response, inputs=[user_input, chat_history], outputs=[chatbot_display, chat_history])

    # Reset button to clear chat history
    reset_button.click(lambda: ([], []), inputs=[], outputs=[chatbot_display, chat_history])

# Launch the Gradio interface
interface.launch()
