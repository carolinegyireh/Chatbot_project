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
model = load_model('./data/model/chatbotmodel.keras')
words = pickle.load(open('./data/model/words.pkl', 'rb'))
classes = pickle.load(open('./data/model/classes.pkl', 'rb'))
with open("./data/data.json", "r") as json_file:
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
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    reply = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(reply) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get the response based on the intent predicted
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if tag in intent['qtype']:
                return random.choice(intent['responses'])
    return "Sorry, this is not in my domain."

# Function that Gradio will use to provide chatbot responses
def chatbot_response(message):
    intents = predict_class(message)
    response = get_response(intents, dict_)
    return response

# Create the Gradio interface
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
