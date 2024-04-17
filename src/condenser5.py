from collections import defaultdict
from heapq import nlargest
from string import punctuation
import math

import streamlit as st
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize



stopwords_ptbr = set(stopwords.words('portuguese')+list(punctuation))
def remove_stop_words_and_punct_in_portuguese(text):
    words= word_tokenize(text.lower())
    return [word for word in words if word not in stopwords_ptbr]

def summarize_text_portuguese(text, n_sent=2):
    words_not_stopwords = remove_stop_words_and_punct_in_portuguese(text)
    sentences = sent_tokenize(text)
    #print(sentences)
    frequency = FreqDist(words_not_stopwords)
    important_sentences = defaultdict(int)
    
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                important_sentences[i] += frequency[word]
                
    numb_sent = n_sent
    idx_important_sentences = nlargest(numb_sent,
                                       important_sentences,
                                       important_sentences.get)
    
    sum = ' '
    for i in sorted(idx_important_sentences):
        sum = sum + sentences[i]+" "
    return sum

def condense5(fulltext):
    k = 0
    for sentence in fulltext.split("."):
        k = k+1
    return summarize_text_portuguese(fulltext,math.ceil(k/5))