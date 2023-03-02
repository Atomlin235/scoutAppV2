
import pandas as pd
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup as bs
import csv
import spacy

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
    #Use Spacy load() to import a model

    nlp = spacy.load('en_core_web_sm')


    # Create EntityRuler pattern matching rules
    tech_skills = [
        {'label': 'SKILL', 'pattern': [{"LOWER": "python"}], 'id': 'python'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "r"}], 'id': 'r'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "sas"}], 'id': 'sas'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "java"}], 'id': 'java'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "excel"}], 'id': 'excel'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "c++"}], 'id': 'c++'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "c#"}], 'id': 'c#'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "c"}], 'id': 'c'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "javascript"}], 'id': 'javascript'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "html"}], 'id': 'html'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "css"}], 'id': 'css'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "php"}], 'id': 'php'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ruby"}], 'id': 'ruby'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "scala"}], 'id': 'scala'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "perl"}], 'id': 'perl'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "matlab"}], 'id': 'matlab'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "hadoop"}], 'id': 'hadoop'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "spark"}], 'id': 'spark'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "hive"}], 'id': 'hive'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "pig"}], 'id': 'pig'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "shark"}], 'id': 'shark'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "oozie"}], 'id': 'oozie'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "zookeeper"}], 'id': 'zookeeper'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "flume"}], 'id': 'flume'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "mahout"}], 'id': 'mahout'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "sqoop"}], 'id': 'sqoop'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "storm"}], 'id': 'storm'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "kafka"}], 'id': 'kafka'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "cassandra"}], 'id': 'cassandra'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "mongodb"}], 'id': 'mongodb'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "redis"}], 'id': 'redis'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "elasticsearch"}], 'id': 'elasticsearch'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "neo4j"}], 'id': 'neo4j'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "sql"}], 'id': 'sql'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "nosql"}], 'id': 'nosql'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "postgresql"}], 'id': 'postgresql'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "oracle"}], 'id': 'oracle'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "mysql"}], 'id': 'mysql'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "sqlite"}], 'id': 'sqlite'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "mariadb"}], 'id': 'mariadb'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "mssql"}], 'id': 'mssql'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "db2"}], 'id': 'db2'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "pandas"}], 'id': 'pandas'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "spacy"}], 'id': 'spacy'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "nltk"}], 'id': 'nltk'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "gensim"}], 'id': 'gensim'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "huggingface"}], 'id': 'huggingface'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "transformers"}], 'id': 'transformers'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "scikit-learn"}], 'id': 'scikit-learn'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "scikit"}, {"LOWER": "learn"}], 'id': 'scikit-learn'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "sklearn"}], 'id': 'scikit-learn'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "tensor"}, {"LOWER": "flow"}], 'id': 'tensorflow'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ux"}, {"LOWER": "design"}], 'id': 'ux design'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ui"}, {"LOWER": "design"}], 'id': 'ui design'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "tensorflow"}], 'id': 'tensorflow'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "keras"}], 'id': 'keras'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "pytorch"}], 'id': 'pytorch'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "numpy"}], 'id': 'numpy'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "scipy"}], 'id': 'scipy'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "matplotlib"}], 'id': 'matplotlib'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "seaborn"}], 'id': 'seaborn'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "plotly"}], 'id': 'plotly'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "bokeh"}], 'id': 'bokeh'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "d3"}], 'id': 'd3'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "airflow"}], 'id': 'airflow'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "docker"}], 'id': 'docker'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "kubernetes"}], 'id': 'kubernetes'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "aws"}], 'id': 'aws'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "amazon"}, {"LOWER": "web"}, {"LOWER": "services"}], 'id': 'aws'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "gcp"}], 'id': 'gcp'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "google"}, {"LOWER": "cloud"}, {"LOWER": "platform"}], 'id': 'gcp'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "azure"}], 'id': 'azure'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "machine learning"}], 'id': 'machine learning'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ml"}], 'id': 'machine learning'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "deep"}, {"LOWER": "learning"}], 'id': 'deep learning'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dl"}], 'id': 'deep learning'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}], 'id': 'nlp'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "nlp"}], 'id': 'nlp'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "computer"}, {"LOWER": "vision"}], 'id': 'computer vision'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "cv"}], 'id': 'computer vision'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "science"}], 'id': 'data science'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ds"}], 'id': 'data science'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "analysis"}], 'id': 'data analysis'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "da"}], 'id': 'data analysis'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "visualisation"}], 'id': 'data visualisation'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "visualization"}], 'id': 'data visualization'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dv"}], 'id': 'data visualisation'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "mining"}], 'id': 'data mining'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dm"}], 'id': 'data mining'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "engineering"}], 'id': 'data engineering'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "de"}], 'id': 'data engineering'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "analytics"}], 'id': 'data analytics'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "da"}], 'id': 'data analytics'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "warehouse"}], 'id': 'data warehouse'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dw"}], 'id': 'data warehouse'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "pipelines"}], 'id': 'data pipelines'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dp"}], 'id': 'data pipelines'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "munging"}], 'id': 'data munging'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dm"}], 'id': 'data munging'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "preparation"}], 'id': 'data preparation'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dp"}], 'id': 'data preparation'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "wrangling"}], 'id': 'data wrangling'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "dw"}], 'id': 'data wrangling'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "data"}, {"LOWER": "cleaning"}], 'id': 'data cleaning'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "kotlin"}], 'id': 'kotlin'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "excel"}], 'id': 'excel'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "open"},{"LOWER": "cv"}], 'id': 'open cv'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "go"}], 'id': 'go'},
    ]

    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(tech_skills)

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

def extract_soft_skills(dataframe):
    #Use Spacy load() to import a model

    nlp = spacy.load('en_core_web_sm')


    # Create EntityRuler pattern matching rules
    soft_skills = [
        {'label': 'SKILL', 'pattern': [{"LOWER": "communication"}], 'id': 'communication'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "teamwork"}], 'id': 'teamwork'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "adaptability"}], 'id': 'adaptability'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "creative"}], 'id': 'creative'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "leadership"}], 'id': 'leadership'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "resourcefulness"}], 'id': 'resourcefulness'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "persuasive"}], 'id': 'persuasive'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "problem"}, {"LOWER": "solving"}], 'id': 'problem-solving'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "time"}, {"LOWER": "management"}], 'id': 'time management'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "descision"}, {"LOWER": "making"}], 'id': 'descision making'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "critical"}, {"LOWER": "thinker"}], 'id': 'critical thinker'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "ambitious"}], 'id': 'ambitious'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "confident"}], 'id': 'confident'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "analytical"}], 'id': 'analytical'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "motivated"}], 'id': 'motivated'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "public"}, {"LOWER": "speaking"}], 'id': 'public speaking'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "innovative"}], 'id': 'innovative'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "respectful"}], 'id': 'respectful'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "detail"}, {"LOWER": "orientated"}], 'id': 'detail orientated'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "interpersonal"}, {"LOWER": "skills"}], 'id': 'interpersonal_skills'},
        {'label': 'SKILL', 'pattern': [{"LOWER": "worker"}], 'id': 'worker'},
    ]



    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(soft_skills)

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

def extract_user_skills(dataframe):

    #Use Spacy load() to import a model

    nlp = spacy.load('en_core_web_sm')


    # Create EntityRuler pattern matching rules
    user_skills = []

    list_of_input_skills = retrieve_skill_list()
    count = 0

    while count < len(list_of_input_skills):
      user_skills.append({'label': 'SKILL', 'pattern': [{"LOWER": list_of_input_skills[count]}], 'id': list_of_input_skills[count]},)
      count = count +1 

    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(user_skills)

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
