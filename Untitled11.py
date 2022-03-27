#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import spacy
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Define english stopwords
stop_words = stopwords.words('english')

# load the spacy module and create a nlp object
# This need the spacy en module to be present on the system.
nlp = spacy.load('en_core_web_sm')
# proces to remove stopwords form a file, takes an optional_word list
# for the words that are not present in the stop words but the user wants them deleted.


def remove_stopwords(text, stopwords=stop_words, optional_params=False, optional_words=[]):
    if optional_params:
        stopwords.append([a for a in optional_words])
    return [word for word in text if word not in stopwords]


def tokenize(text):
    # Removes any useless punctuations from the text
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text)


def lemmatize(text):
    # the input to this function is a list
    str_text = nlp(" ".join(text))
    lemmatized_text = []
    for word in str_text:
        lemmatized_text.append(word.lemma_)
    return lemmatized_text

# internal fuction, useless right now.


def _to_string(List):
    # the input parameter must be a list
    string = " "
    return string.join(List)


def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Takes in Tags which are allowed by the user and then elimnates the rest of the words
    based on their Part of Speech (POS) Tags.
    """
    filtered = []
    str_text = nlp(" ".join(text))
    for token in str_text:
        if token.pos_ in postags:
            filtered.append(token.text)
    return filtered


# In[2]:


import spacy
#import Distill
 
try:
    nlp = spacy.load('en_core_web_sm')

except ImportError:
    print("Spacy's English Language Modules aren't present \n Install them by doing \n python -m spacy download en_core_web_sm")


def _base_clean(text):
    """
    Takes in text read by the parser file and then does the text cleaning.
    """
    text = tokenize(text)
    text = remove_stopwords(text)
    text = remove_tags(text)
    text = lemmatize(text)
    return text


def _reduce_redundancy(text):
    """
    Takes in text that has been cleaned by the _base_clean and uses set to reduce the repeating words
    giving only a single word that is needed.
    """
    return list(set(text))


def _get_target_words(text):
    """
    Takes in text and uses Spacy Tags on it, to extract the relevant Noun, Proper Noun words that contain words related to tech and JD. 

    """
    target = []
    sent = " ".join(text)
    doc = nlp(sent)
    for token in doc:
        if token.tag_ in ['NN', 'NNP']:
            target.append(token.text)
    return target


# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
# https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05

def Cleaner(text):
    sentence = []
    sentence_cleaned = _base_clean(text)
    sentence.append(sentence_cleaned)
    sentence_reduced = _reduce_redundancy(sentence_cleaned)
    sentence.append(sentence_reduced)
    sentence_targetted = _get_target_words(sentence_reduced)
    sentence.append(sentence_targetted)
    return sentence


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer


def do_tfidf(token):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=0.002)
    words = tfidf.fit_transform(token)
    sentence = " ".join(tfidf.get_feature_names())
    return sentence


# In[4]:


from operator import index
from pandas._config.config import options
import textract as tx
import pandas as pd
import os


resume_dir = "Data/Resumes/"
job_desc_dir = "Data/JobDesc/"
resume_names = os.listdir(resume_dir)
job_description_names = os.listdir(job_desc_dir)

document = []


def read_resumes(list_of_resumes, resume_directory):
    placeholder = []
    for res in list_of_resumes:
        temp = []
        temp.append(res)
        text = tx.process(resume_directory+res, encoding='ascii')
        text = str(text, 'utf-8')
        temp.append(text)
        placeholder.append(temp)
    return placeholder


document = read_resumes(resume_names, resume_dir)


def get_cleaned_words(document):
    for i in range(len(document)):
        raw = Cleaner(document[i][1])
        document[i].append(" ".join(raw[0]))
        document[i].append(" ".join(raw[1]))
        document[i].append(" ".join(raw[2]))
        sentence = do_tfidf(document[i][3].split(" "))
        document[i].append(sentence)
    return document


Doc = get_cleaned_words(document)

Database = pd.DataFrame(document, columns=[
                        "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])

Database.to_csv("Resume_Data.csv", index=False)

# Database.to_json("Resume_Data.json", index=False)


def read_jobdescriptions(job_description_names, job_desc_dir):
    placeholder = []
    for tes in job_description_names:
        temp = []
        temp.append(tes)
        text = tx.process(job_desc_dir+tes, encoding='ascii')
        text = str(text, 'utf-8')
        temp.append(text)
        placeholder.append(temp)
    return placeholder


job_document = read_jobdescriptions(job_description_names, job_desc_dir)

Jd = get_cleaned_words(job_document)

jd_database = pd.DataFrame(Jd, columns=[
                           "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])

jd_database.to_csv("Job_Data.csv", index=False)


# In[5]:


import textdistance as td


def match(resume, job_des):
    j = td.jaccard.similarity(resume, job_des)
    s = td.sorensen_dice.similarity(resume, job_des)
    c = td.cosine.similarity(resume, job_des)
    o = td.overlap.normalized_similarity(resume, job_des)
    total = (j+s+c+o)/4
    # total = (s+o)/2
    return total*100


# In[ ]:





# In[36]:


def calculate_scores(resumes, job_description):
    scores = []
    for x in range(resumes.shape[0]):
        score = Similar.match(resumes['TF_Based'][x], job_description['TF_Based'][index])
        scores.append(score)
    return scores


# In[37]:


import Similar


# In[38]:


Resumes = pd.read_csv('Resume_Data.csv')
Jobs = pd.read_csv('Job_Data.csv')


# In[39]:


index = len(Jobs['Name'])-1


# In[40]:


Resumes.shape


# In[41]:


Resumes['Scores'] = calculate_scores(Resumes, Jobs)


# In[42]:


Resumes['Scores']


# In[ ]:




