import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import string 
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup as bs
from csv import DictWriter

import skill_mod
import named_entity_rec
#python -m textblob.download_corpora If required

# 1. Request data from the `devitjobs.uk` API and make queryable using `BeautifulSoup`.

URL = "https://devitjobs.uk/job_feed.xml"
page = requests.get(URL)

soup = bs(page.text, "xml") 

# 2. Make a dataframe containing the relevant data (with `pandas`).

titles = soup.find_all('title')
salary = soup.find_all('salary')
desc = soup.find_all('description')
jobs = soup.find_all('job')

data = []
for i in range(0,len(jobs)):
   rows = [jobs[i].id,titles[i].get_text(),
           salary[i].get_text(),desc[i].get_text()]
   data.append(rows)

df = pd.DataFrame(data,columns = ['Id','Title',
                                  'Salary','Description'], dtype="string")

#cleaning data
initial_df = df


pattern = r'\[[^()]*\]'

initial_df["MinSalary"] = np.nan
initial_df["MaxSalary"] = np.nan

for index, row in initial_df.iterrows():
    # get id text
    #print(row['Id'])
    id_soup = bs(row['Id'],features="lxml")
    id = id_soup.get_text()
    #id = bs(row['Id'],features="lxml").get_text()

    # remove salary from title
    title = re.sub(pattern, '', row['Title'])

    # get description text
    description = bs(row['Description'],features="lxml").get_text()

    # split salary
    salary_text = re.findall("[\$0-9,\. ]*-[\$0-9,\. ]*", row['Salary'])
    _range_list = re.split("-", salary_text[0])
    
    range_list = []

    for salary in _range_list:
        range_list.append(locale.atof(salary))

    # assign
    initial_df.loc[index, "Id"] = id

    initial_df.loc[index, "Title"] = title

    initial_df.loc[index, "Description"] = description
    initial_df.loc[index, "Salary"] = str((range_list[0] + range_list[1])/2)
    initial_df.loc[index, "MinSalary"] = range_list[0]
    initial_df.loc[index, "MaxSalary"] = range_list[1]
    

clean_data_df  = initial_df
clean_data_df['Salary'] = clean_data_df['Salary'].astype(float)