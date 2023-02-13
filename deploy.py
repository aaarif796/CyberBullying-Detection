# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:51:26 2023

@author: aaari
"""
import re
import unidecode
from joblib import load
import streamlit as st
model=load('Gradient.joblib')
tfidf=load('Tfidf.joblib')
pca=load('pca.joblib')
lenc=load('len.joblib')


# s1="I am very bad i like to kill people i like to kill muslim"



def clean_text(s1):
    s1=s1.lower()
    CONTRACTION_MAP={
        "ain't":"is not",
        "aren't":"are not",
        "can't":"cannot",
        "can't've":"cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    string=""
    for word in s1.split(" "):
        if word.strip() in list(CONTRACTION_MAP.keys()):
            string=string+" "+CONTRACTION_MAP[word]
        else:
            string=string+" "+word
    #s1.append(string.strip())
    s1=string
    s1=re.sub(r'http\S+','',s1)
    s1=re.sub(r'\ [A-Za-z]*\.com'," ",s1)
    s1=unidecode.unidecode(s1)
    s1=re.sub(r"\s+"," ",s1)
    s1=re.sub(r"[^a-zA-Z]"," ",s1)
    s1=re.sub(r"\s+"," ",s1)
    s1=s1.strip()
    return s1

# s1=[clean_text(s1)]
# s1=tfidf.transform(s1).toarray()
# s1_t=pca.transform(s1)
# ab=model.predict(s1_t)[0]
# print(f"{lenc.classes_[ab]}")


def app():
    st.set_page_config("dark")
    st.title("CyberBullying Prediction App")
    inputs=st.text_input("Enter the messages:")
    if st.button("Predict"):
        if len(inputs)<=10:
            st.write("Please enter long message.")
        else:
            inputs=[clean_text(inputs)]
            inputs=tfidf.transform(inputs).toarray()
            inputs=pca.transform(inputs)
            a=model.predict(inputs)[0]
            pred=lenc.classes_[a]
            st.write(f"The prediction is {pred}")
            st.success("Thank You")
     
    
if __name__=="__main__":
    app()