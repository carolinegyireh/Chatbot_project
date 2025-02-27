#  Medical Chatbot
### Overview
In this project, I developed a **Medical Chatbot** using **MedQuad dataset (Medical Question-Answer)** from [kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/datahttps://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/data). The dataset includes questions about treatments, chronic diseases, medical protocols, and answers from healthcare professionals such as doctors, nurses, and pharmacists.
The aim of the project is to assist patients by providing reliable medical information and supporting healthcare professionals by automating responses to medical queries the chatbot is trained with.

### Dataset
The Dataset was categorized into 3 columns:

1. **qtype**: The medical category of the question.

2. **Question:** The patient's medical inquiry.

3. **Answer:** The response from healthcare professionals.

and 16407  rows with no data missing. 

### Requirements
Python 3.7 or higher
Required libraries:
- Numpy
- pandas
- random
- json
- pickle
- torch
- gradio
- transformers
- tensorflow
- nltk

You can install these libraries using the command:
```pip install -r requirements.txt```

## Setup
Clone the Repository:
git clone  ``` https://github.com/cgyireh1/Chatbot_project.git```


Navigate to the cloned directory: ```cd Chatbot_project```


Run ```pip install -r requirements.txt``` to install all the required libraries: 

## Preprocessing And Model Training
- Tokenization: Tokenizing and lemmatizing the input text to reduce vocabulary complexity.
- Normalization: Removing unnecessary characters and converting to lowercase.
- Formatting: Converting the dataset is in the JSON format for training

Model Training
- Embedding Generation: Use BERT embeddings for sentence-level understanding.
- Fine-tuning the Model: Fine-tune a pre-trained Transformer model, BERT on the dataset for better conversational response.
- Hyperparameter Tuning: Experimented with learning rates, batch size, epochs, and other hyperparameters to optimize performance.
- BERT tokenizer and model are initialized to get embeddings for the user input.


## How to Run The Chatbot

- To run the chatbot on terminal, use the command:

Run ```python chatbot-gracio.py```

 The Gradio interface is launched and a web page will open in your default browser.
  
  Type your medical query into the text box and click the submit button.
  
  The chatbot will respond with the required medical information.

- To run the chatbot on the colab:

Run the cell with the gradio interface
           The Gradio interface is launched.
           
Type your medical query into the text box and click the submit button.

The chatbot will respond with the required medical information.

## Performance metrics
```
Accuracy: 0.9993
F1 Score: 0.9992
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       727
           1       1.00      0.96      0.98        46
           2       1.00      1.00      1.00       235
           3       1.00      1.00      1.00       653
           4       1.00      1.00      1.00      1120
           5       1.00      1.00      1.00      1087
           6       1.00      1.00      1.00      4535
           7       1.00      1.00      1.00      1446
           8       1.00      1.00      1.00       361
           9       0.99      0.99      0.99       210
          10       1.00      1.00      1.00       395
          11       1.00      1.00      1.00        77
          12       1.00      0.00      0.00         1
          13       0.99      1.00      1.00       324
          14       1.00      1.00      1.00      2748
          15       1.00      1.00      1.00      2442

    accuracy                           1.00     16407
   macro avg       1.00      0.93      0.93     16407
weighted avg       1.00      1.00      1.00     16407
```

