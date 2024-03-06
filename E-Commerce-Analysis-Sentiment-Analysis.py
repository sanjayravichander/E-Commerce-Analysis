#!/usr/bin/env python
# coding: utf-8

"""
Created on Tue Mar  5 21:46:17 2024

@author: DELL
"""
# Sentiment Analysis

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Importing the Data
df =pd.read_json("C:\\Users\\DELL\\Downloads\\Tableau Project\\E-Commerce Analysis\\fashion_products_dataset.json")
df.isnull().sum()

text=df['description']

## Text Prepreocessing
#Converting to lower
text=text.str.lower()

##Remove spl char and numbers
text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

## Tokenization
text=text.apply(word_tokenize)

## Removing stop words
stop_words=set(stopwords.words('english'))
text = text.apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
#porter = PorterStemmer()
#text = text.apply(lambda x: [porter.stem(word) for word in x])

lemmatizer = WordNetLemmatizer()
text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert back to text
text = text.apply(lambda x: ' '.join(x))

## Converting series to DataFrame
text = pd.DataFrame({'Preprocessed_Text': text})

#Concatinating the preprocessed text column along with df
df = pd.concat([df, text], axis=1)

## Dropping the non preprocessed ['Review_Text'] column
df=df.drop(columns=['description'])

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from torch.utils.data import Dataset
import torch

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        
        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

# Assuming texts is your list of descriptions and labels are integers 0 (negative), 1 (neutral), 2 (positive)
texts = df['Preprocessed_Text']
labels = df['average_rating']  # Example labels

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = SentimentDataset(texts, labels, tokenizer)

# Load DistilBERT model with 3 expected classes
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define TrainingArguments and Trainer as before, then train your model.

# Prediction and Labeling
def predict_sentiment(texts, model, tokenizer):
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predictions.append(predicted_class_id)
    return predictions

# Convert numeric predictions to labels
def numeric_to_labels(predictions):
    label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [label_dict[pred] for pred in predictions]

# Example usage
predictions = predict_sentiment(texts, model, tokenizer)
sentiment_labels = numeric_to_labels(predictions)

# Example usage
predictions = predict_sentiment(texts, model, tokenizer)
sentiment_labels = numeric_to_labels(predictions)1`2aQQ


# In[4]:


df['sentiment_labels']=sentiment_labels


# In[5]:


import nltk
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Assuming df is your DataFrame and it has columns 'description' and 'Sentiment_Label'
# Example: df = pd.DataFrame({'description': ['love this movie', 'hate this movie'], 'Sentiment_Label': ['positive', 'negative']})

# Function to get bigrams
def get_bigrams(texts):
    bigrams = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        bigram = list(ngrams(tokens, 2))
        bigrams.extend(bigram)
    return [" ".join(bigram) for bigram in bigrams]

# Generate word cloud
def generate_wordcloud(bigrams, title):
    bigram_counts = Counter(bigrams)
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(bigram_counts)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Separating positive and negative descriptions
positive_texts = df[df['sentiment_labels'] == 'positive']['Preprocessed_Text']
negative_texts = df[df['sentiment_labels'] == 'neutral']['Preprocessed_Text']

# Getting bigrams
positive_bigrams = get_bigrams(positive_texts)
negative_bigrams = get_bigrams(negative_texts)

# Generating word clouds
generate_wordcloud(positive_bigrams, 'Positive Sentiment Bigrams')
generate_wordcloud(negative_bigrams, 'Neutral Sentiment Bigrams')

#Saving dataset
df.to_json('E-Commerce.json')


# In[8]:


df.to_excel('E-Commerce-Analysis.xlsx')


# In[9]:


df.head()


# In[50]:


df['product_details']#[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


df=pd.read_excel("C:\\Users\\DELL\\Downloads\\E-Commerce-Analysis.xlsx")


# In[88]:


df.head()


# In[89]:


df.shape


# In[90]:


df['product_details'][0]


# In[91]:


import pandas as pd
import ast

# Convert the string representation of list to actual list
df['product_details'] = df['product_details'].apply(ast.literal_eval)

# Prepare a list to hold all the new data
new_data_list = []

# Create new columns from the list of dictionaries
for index, row in df.iterrows():
    new_row = {}
    for dict in row['product_details']:
        for key in dict:
            new_row[key] = dict[key]
    new_data_list.append(new_row)

# Convert the list of dictionaries into a DataFrame
new_data = pd.DataFrame(new_data_list)

# Concatenate the original DataFrame with the new DataFrame
df = pd.concat([df, new_data], axis=1)

# Now, you can drop the original 'product_details' column
df = df.drop(columns=['product_details'])


# In[92]:


df


# In[93]:


df.shape


# In[94]:


# Calculate the null percentage for each column
null_percentage = df.isnull().sum() / len(df) * 100

# Get the columns to drop
columns_to_drop = null_percentage[null_percentage > 50]
columns_to_drop


# In[95]:


# Calculate the null percentage for each column
null_percentage = df.isnull().sum() / len(df) * 100

# Get the columns to drop
columns_to_drop = null_percentage[null_percentage > 50].index

# Drop the columns
df = df.drop(columns=columns_to_drop)


# In[97]:


df.shape


# In[98]:


df


# In[99]:


df.to_excel('E-Commerce-Analysis-Full.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




