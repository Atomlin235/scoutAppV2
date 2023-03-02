import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
from babel.numbers import parse_decimal
import string 
import locale
locale.setlocale(locale.LC_ALL, 'C')
import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup as bs
from csv import DictWriter

#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import named_entity_rec
#python -m textblob.download_corpora If required

# 1. Request data from the `devitjobs.uk` API and make queryable using `BeautifulSoup`.

URL = "https://devitjobs.uk/job_feed.xml"
page = requests.get(URL)

soup = bs(page.text, "lxml-xml") 

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
        range_list.append(float(parse_decimal(salary, locale="en_US")))

    # assign
    initial_df.loc[index, "Id"] = id

    initial_df.loc[index, "Title"] = title

    initial_df.loc[index, "Description"] = description
    initial_df.loc[index, "Salary"] = str((range_list[0] + range_list[1])/2)
    initial_df.loc[index, "MinSalary"] = range_list[0]
    initial_df.loc[index, "MaxSalary"] = range_list[1]
    

clean_data_df  = initial_df
clean_data_df['Salary'] = clean_data_df['Salary'].astype(float)
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
def main():
    menu =["Title Search","Description Search"]
    with st.sidebar:
      #  url_search = st.text_input("//TODO Database URL")
        st.subheader("Main Menu")
        choice = st.selectbox("Menu List",menu)

        st.subheader("User skills")
        search_term = st.text_input("Enter skill")
        button_clicked = st.button("OK")

        if button_clicked:
            dict = {'Technical_Skills': search_term}

            field_names = ['Technical_Skills']
            with open('technical_skills.csv', 'a',newline='') as f_object:
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
                dictwriter_object.writerow(dict)
                f_object.close()

        #with st.expander("See explanation"):
        df = pd.read_csv('technical_skills.csv')

        # Sidebar - Sector selection
        sorted_sector_unique = sorted( df['Technical_Skills'].unique() )
        selected_sector = st.sidebar.multiselect('Technical_Skills', sorted_sector_unique, sorted_sector_unique)

        # Filtering data
        df_selected_sector = df[ (df['Technical_Skills'].isin(selected_sector)) ]
        df_selected_sector.to_csv('technical_skills.csv',index=False)
       
    st.title("Scout Job Search")

    if choice == "Title Search":
        st.subheader("Title Search")

        with st.form(key='searchform'):
            nav1,nav2= st.columns([2,1])

            with nav1:
                search_term = st.text_input("Search by Title")
            with nav2:
                st.text("Search")
                submit_search = st.form_submit_button(label="Search")
        
    # Title Data Frame
      # Title Data Frame
        if submit_search:
            st.success ("You searched for {} jobs".format(search_term))
            title_search_df = clean_data_df[clean_data_df.Title.str.contains(pat=search_term,case=False)]
            num_results = len(title_search_df.index)
            st.subheader("Showing {} jobs with {} in the title".format(num_results,search_term))
            simplified_title_search = title_search_df.iloc[:, [1,2,3,4,5]].copy()
            st.dataframe(simplified_title_search)

      #General Top skill extraction
            st.subheader("NER Skill Extraction Tool")
            ner_title_df = named_entity_rec.extract_tech_skills(title_search_df)
            ner_tech_skill_df = ner_title_df[['Title','Skills']].copy()
            ner_soft_skills_df = named_entity_rec.extract_soft_skills(title_search_df)
            ner_soft_skills_df.rename(columns={'Skills': 'SoftSkills'}, inplace=True)
            ner_skill_ex_df = ner_soft_skills_df[["SoftSkills",'Salary']].copy()
            
            df_top_combined = pd.concat([ner_tech_skill_df,ner_skill_ex_df], axis = 1)
            st.dataframe(df_top_combined)

         # Most imporant skills
            
            st.header("The most important technical skills for {} jobs ".format(search_term))
            df_skills = df_top_combined.explode('Skills')
            df_summary = df_skills.groupby('Skills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            df_sal_summary= df_skills.groupby('Skills').agg(roles=('Title', 'count'),min_salary=('Salary', 'min'),med_salary=('Salary','median'),max_salary=('Salary', 'max'),).sort_values('roles', ascending=False)
            
    
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skill to roles")
                st.bar_chart(df_summary)

            with col2:
                st.subheader("Skill to salary potential for {} jobs".format(search_term))
                st.bar_chart(df_sal_summary)

            with st.expander("For information on the above data"):
                st.dataframe(df_sal_summary)

            st.subheader("The most important soft skills for {} jobs".format(search_term))
            df_soft_skills = df_top_combined.explode('SoftSkills')
            df_soft_summary = df_soft_skills.groupby('SoftSkills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            st.bar_chart(df_soft_summary)
            with st.expander("See explanation"):
                st.dataframe(df_soft_summary)



            


    # NER MISSING SKIILS 
            st.subheader("NER Missing Skill Extraction Tool")
            st.subheader("Missing Skills for {} jobs".format(search_term))
            ner_user_df = named_entity_rec.extract_user_skills(title_search_df)

            #setting up general skills
            ner_tech_skill_df.rename(columns={'Skills': 'technical_skills'}, inplace=True)
            df_general_desc_skills = ner_tech_skill_df[['Title','technical_skills']].copy()
            df_general_desc_skills.rename(columns={'technical_skills': 'general_skills'}, inplace=True)
   

            #setting up user skilss
            ner_user_skills_df = ner_user_df[['Title','Skills']].copy()
            ner_user_skills_df.rename(columns={'Skills': 'technical_skills'}, inplace=True)
            df_users_skills = ner_user_skills_df['technical_skills']
            
            df_combined = pd.concat([df_general_desc_skills,df_users_skills], axis = 1)

            df_extract = df_combined[['general_skills','technical_skills']]
            temp = df_extract[['general_skills','technical_skills']].applymap(set)
            temp.rename(columns={'general_skills': 'missing_skills'}, inplace=True)
            missing = temp.diff(periods=-1, axis=1).dropna(axis=1) 

            df_missing_skill = pd.concat([df_combined ,missing], axis = 1)
            df_missing_skill.rename(columns={'technical_skills': 'user_skills'}, inplace=True)
            
            st.dataframe(df_missing_skill)




            # Most imporant skills GRAPH S NEEDED 
            st.subheader("The most important missing skills for {} jobs".format(search_term))
            df_missing_skills = df_missing_skill.explode('missing_skills')
            df_missing_summary = df_missing_skills.groupby('missing_skills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            st.bar_chart(df_missing_summary)
            with st.expander("See explanation"):
                st.dataframe(df_missing_summary)

            

            # replace empty strings with null and drop 
             #df_combined['Skills'].replace('', np.nan, inplace=True)
           # df_combined(subset=['Tenant'], inplace=True)
          

 #User Missing Skill Extraction
            
            

    elif choice == "Description Search":
        st.subheader("Description Search")

        with st.form(key='searchform'):
            nav1,nav2,= st.columns([2,1])

            with nav1:
                search_term = st.text_input("Search by Description")
            
            with nav2:
                st.text("Search")
                submit_search = st.form_submit_button(label="Search")

        if submit_search:
            st.success ("You searched for {} jobs".format(search_term))
            description_search_df = clean_data_df[clean_data_df.Description.str.contains(pat=search_term,case=False)]
            num_results = len(description_search_df.index)
            st.subheader("Showing {} jobs with {} in the description".format(num_results,search_term))
            simplified_desc_search = description_search_df.iloc[:, [1,2,3,4,5]].copy()
            st.dataframe(simplified_desc_search)
            
            st.subheader("NER Skill Extraction Tool")
            ner_desc_df = named_entity_rec.extract_tech_skills(description_search_df)
            ner_tech_skill_df = ner_desc_df[['Title','Skills']].copy()
            ner_soft_skills_df = named_entity_rec.extract_soft_skills(description_search_df)
            ner_soft_skills_df.rename(columns={'Skills': 'SoftSkills'}, inplace=True)
            ner_skill_ex_df = ner_soft_skills_df[["SoftSkills","Salary"]].copy()
            
            df_top_combined = pd.concat([ner_tech_skill_df,ner_skill_ex_df], axis = 1)
            st.dataframe(df_top_combined)

            # Most imporant skills GRAPH S NEEDED 

            st.header("The most important technical skills for {} jobs ".format(search_term))
            df_skills = df_top_combined.explode('Skills')
            df_summary = df_skills.groupby('Skills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            df_sal_summary= df_skills.groupby('Skills').agg(roles=('Title', 'count'),min_salary=('Salary', 'min'),med_salary=('Salary','median'),max_salary=('Salary', 'max'),).sort_values('roles', ascending=False)
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skill to roles")
                st.bar_chart(df_summary)

            with col2:
                st.subheader("Skill to salary potential for {} jobs".format(search_term))
                st.bar_chart(df_sal_summary)

            with st.expander("For information on the above data"):
                st.dataframe(df_sal_summary)

           

            st.subheader("The most important soft skills for {} jobs".format(search_term))
            df_soft_skills = df_top_combined.explode('SoftSkills')
            df_soft_summary = df_soft_skills.groupby('SoftSkills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            st.bar_chart(df_soft_summary)
            with st.expander("See explanation"):
                st.dataframe(df_soft_summary)
            

            st.subheader("NER Missing Skill Extraction Tool")
            st.subheader("Missing Skills for {} jobs".format(search_term))
            ner_user_df = named_entity_rec.extract_user_skills(description_search_df)
            
            #setting up general skills
            ner_tech_skill_df.rename(columns={'Skills': 'technical_skills'}, inplace=True)
            df_general_desc_skills = ner_tech_skill_df[['Title','technical_skills']].copy()
            df_general_desc_skills.rename(columns={'technical_skills': 'general_skills'}, inplace=True)



            #setting up user skilss
            ner_user_skills_df = ner_user_df[['Title','Skills']].copy()
            ner_user_skills_df.rename(columns={'Skills': 'technical_skills'}, inplace=True)
            df_users_skills = ner_user_skills_df['technical_skills']
            
            df_combined = pd.concat([df_general_desc_skills,df_users_skills], axis = 1)

            df_extract = df_combined[['general_skills','technical_skills']]
            temp = df_extract[['general_skills','technical_skills']].applymap(set)
            temp.rename(columns={'general_skills': 'missing_skills'}, inplace=True)
            missing = temp.diff(periods=-1, axis=1).dropna(axis=1) 

            df_missing_skill = pd.concat([df_combined ,missing], axis = 1)
            df_missing_skill.rename(columns={'technical_skills': 'user_skills'}, inplace=True)
            
            st.dataframe(df_missing_skill)

# Most imporant skills GRAPH S NEEDED 
            st.subheader("The most important missing skills for {} jobs".format(search_term))
            df_missing_skills = df_missing_skill.explode('missing_skills')
            df_missing_summary = df_missing_skills.groupby('missing_skills').agg(roles=('Title', 'count'),).sort_values('roles', ascending=False)
            st.bar_chart(df_missing_summary)
            with st.expander("See explanation"):
                st.dataframe(df_missing_summary)


           

  
if __name__ == "__main__":
    main()
