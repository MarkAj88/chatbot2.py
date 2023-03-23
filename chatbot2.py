#!/usr/bin/env python
# coding: utf-8

# In[2]:

requirements.txt
!pip install scikit-learn

import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join(text.split())
    return text

# Function to get the most relevant sentence
def get_most_relevant_recipe(query, sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_sentences = tfidf_vectorizer.fit_transform(sentences)
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_sentences)
    most_similar_sentence = sentences[similarity_scores.argmax()]
    return most_similar_sentence

# Chatbot function
def chatbot(query):
    with open("chatbot.txt", "r", encoding="utf-8") as f:
        text = f.read()
    sentences = text.split("\n")
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    most_relevant_recipe = get_most_relevant_recipe(preprocess(query), preprocessed_sentences)
    return most_relevant_recipe

# Streamlit app
def main():
    st.title("Cooking Chatbot")
    query = st.text_input("What would you like to cook today?")
    if query:
        response = chatbot(query)
        st.write(response)

if __name__ == "__main__":
    main()


# In[ ]:




