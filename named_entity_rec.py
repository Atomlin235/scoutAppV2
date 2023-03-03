
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup as bs
import csv
import spacy
nlp = spacy.load('en_core_web_sm')

#python -m spacy download en_core_web_sm

#python -m textblob.download_corpora If required
def retrieve_skill_list():
    file = open("technical_skills.csv", "r", encoding="utf8")
    raw_technical_skills = list(csv.reader(file))
    joint_skills = list(map(''.join, raw_technical_skills))
    technical_skills = list(map(lambda x: x.lower(), joint_skills))
    file.close()
    return technical_skills



def extract_tech_skills(dataframe):

    # Cleaning of data Convert all text to lower cases Delete all tabulation,spaces, and new lines, Delete all numericals Delete nltk's defined stop words,Lemmatize text   
    dataframe ['Description'] = dataframe['Description'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
    dataframe ['Description'] = dataframe ['Description'].str.replace('[^\w\s]',' ')

    #Extract the skills to a new column
    dataframe['Skills'] = dataframe['Description'].apply(lambda x: [ent.ent_id_ for ent in nlp(x).ents if ent.label_ == 'SKILL'])


    # use another lambda function to use set() to de-duplicate the values and return only the unique matches in a Python list
    dataframe['Skills'] = dataframe['Skills'].apply(lambda x: list(set(x)))

    #Use the named entities to clean the dataset

    dataframe[['Title', 'Skills']].sort_values('Skills', key=lambda x: x.str.len(), ascending=True).head(100)
    
    return dataframe

def extract_soft_skills(dataframe):
    # Cleaning of data Convert all text to lower cases Delete all tabulation,spaces, and new lines, Delete all numericals Delete nltk's defined stop words,Lemmatize text   
    dataframe ['Description'] = dataframe['Description'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
    dataframe ['Description'] = dataframe ['Description'].str.replace('[^\w\s]',' ')

    #Extract the skills to a new column
    dataframe['Skills'] = dataframe['Description'].apply(lambda x: [ent.ent_id_ for ent in nlp(x).ents if ent.label_ == 'SOFT_SKILL'])


    # use another lambda function to use set() to de-duplicate the values and return only the unique matches in a Python list
    dataframe['Skills'] = dataframe['Skills'].apply(lambda x: list(set(x)))

    #Use the named entities to clean the dataset

    dataframe[['Title', 'Skills']].sort_values('Skills', key=lambda x: x.str.len(), ascending=True).head(100)
    
    return dataframe

def extract_user_skills(dataframe):

    #Use Spacy load() to import a model

    nlp = spacy.load('en_core_web_sm')


    # Create EntityRuler pattern matching rules
    user_skills = []

    list_of_input_skills = retrieve_skill_list()
    count = 0

    while count < len(list_of_input_skills):
      user_skills.append({'label': 'USER_SKILL', 'pattern': [{"LOWER": list_of_input_skills[count]}], 'id': list_of_input_skills[count]},)
      count = count +1 

    user_ruler = nlp.add_pipe('entity_ruler', 'user_ruler')
    user_ruler.add_patterns(user_skills)

    #Clean incoming data 

    # Cleaning of data Convert all text to lower cases Delete all tabulation,spaces, and new lines, Delete all numericals Delete nltk's defined stop words,Lemmatize text   
    dataframe ['Description'] = dataframe['Description'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
    dataframe ['Description'] = dataframe ['Description'].str.replace('[^\w\s]',' ')

    #Extract the skills to a new column
    dataframe['Skills'] = dataframe['Description'].apply(lambda x: [ent.ent_id_ for ent in nlp(x).ents if ent.label_ == 'SKILL'])


    # use another lambda function to use set() to de-duplicate the values and return only the unique matches in a Python list
    dataframe['Skills'] = dataframe['Skills'].apply(lambda x: list(set(x)))

    #Use the named entities to clean the dataset

    dataframe[['Title', 'Skills']].sort_values('Skills', key=lambda x: x.str.len(), ascending=True).head(100)
    #title_skills_df =dataframe[['Title', 'Skills']].copy()
  #  st.dataframe(title_skills_df)


    #Analyse the distribution of named entities

   # df_skills = dataframe.explode('Skills')

   # df_summary = df_skills.groupby('Skills').agg(
   #     roles=('Title', 'count'),
        
  #  ).sort_values('roles', ascending=False)

   # st.dataframe(df_summary)
    
    return dataframe
