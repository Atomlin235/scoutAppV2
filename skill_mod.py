
import pandas as pd
import numpy as np
import re
import string 
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from textblob import Word
from textblob import TextBlob





def retrieve_skill_list():
    file = open("technical_skills.csv", "r", encoding="utf8")
    raw_technical_skills = list(csv.reader(file))
    joint_skills = list(map(''.join, raw_technical_skills))
    technical_skills = list(map(lambda x: x.lower(), joint_skills))
    file.close()
    return technical_skills
def retrieve_general_skill_list():
    file = open("general_technical_skills.csv", "r", encoding="utf8")
    raw_technical_skills = list(csv.reader(file))
    joint_skills = list(map(''.join, raw_technical_skills))
    technical_skills = list(map(lambda x: x.lower(), joint_skills))
    file.close()
    return technical_skills

def user_spec_key_skills(dataframe):

    # Dataframe from title and Description Column
    title_desc_df = dataframe.iloc[:, [1, 3]].copy()

# Cleaning of data Convert all text to lower cases Delete all tabulation,spaces, and new lines, Delete all numericals Delete nltk's defined stop words,Lemmatize text   
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
    title_desc_df ['Description'] = title_desc_df ['Description'].str.replace('[^\w\s]',' ')
    ## digits
    title_desc_df ['Description'] = title_desc_df ['Description'].str.replace('\d+', '')

    #remove stop words
    stop = stopwords.words('english')
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            ## lemmatization
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
         #--   st.dataframe(title_desc_df)

        # remove further meaningless words PERHAPS CSV OF STOP WORDS
       
    other_stop_words = ['gbp','requirement','required','per','year','junior', 'senior','experience','etc','job','work','company','technique',
                    'candidate','skill','skills','language','menu','inc','new','plus','years',
                   'technology','organization','ceo','cto','account','manager','scientist','mobile',
                    'developer','product','revenue','strong','full','salary']

    title_desc_df['Description'] = title_desc_df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stop_words))
          #--  st.dataframe(title_desc_df)

        # train the naive bayes algorithm.

        ## Converting text to features 
    vectorizer = TfidfVectorizer()
        #Tokenize and build vocabulary
    X = vectorizer.fit_transform(title_desc_df.Description)
    y = title_desc_df.Title

        # split data into 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
          #--  st.write("train data shape: ",X_train.shape)--
          #--  st.write("test data shape: ",X_test.shape)--

        # Fit model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
         ## Predict
    y_predicted = clf.predict(X_test)

     #------------------------------------------Model Evaluation-----------------------------------
        #evaluate the predictions
         # --  st.write("Accuracy score is: ",accuracy_score(y_test, y_predicted))
          # -- st.write("Classes: (to help read Confusion Matrix)\n", clf.classes_)
          #  --st.write("Confusion Matrix: ")
          #  --st.write(confusion_matrix(y_test, y_predicted))
          # -- st.write("Classification Report: ")
          # -- st.write(classification_report(y_test, y_predicted))
    #---------------------------------------------------------------------------------------------

    # Output At this step, we have for each class/job a list of the most representative words/tokens found in job descriptions.
    #shrink to list of words to only:
    #5 technical skills 5 adjectives


    # //TO_DO Aattempt at using a CSV Solution does not work 
    # file = open("technical_skills.csv", "r", encoding="utf8")
    # raw_technical_skills = list(csv.reader(file))
    # joint_skills = list(map(''.join, raw_technical_skills))
    # technical_skills = list(map(lambda x: x.lower(), joint_skills))
    # file.close()
    technical_skills = retrieve_skill_list()
    soft_skills = ['communication', 'teamwork','problem solving', 'time management','critical thinking','decision-making','organizational','stress management','leadership','creative','ambitious',
                   'determined','resourceful','persuasive','public speaking','responsible','hard working']
    
    
    feature_array = vectorizer.get_feature_names()
            # number of overall model features
    features_numbers = len(feature_array)
## max sorted features number
    n_max = int(features_numbers * 0.1)

##initialize output dataframe
    output = pd.DataFrame()
    for i in range(0,len(clf.classes_)):
        print("\n****" ,clf.classes_[i],"****\n")
        class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
        raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    ## Extract technical skills
        top_technical_skills= list(set(technical_skills).intersection(raw_skills))[:5]
    ## Soft Skills
        top_soft_skills = list(set(soft_skills).intersection(raw_skills))[:5]

    # transform list to string
        txt = " ".join(raw_skills)
        blob = TextBlob(txt)
    #top 5 adjective
        top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:5]
    #print("Top 5 : top_technical_skills",top_soft_skills)
    
        output = output.append({'Title':clf.classes_[i],
                        'technical_skills':top_technical_skills,
                        'soft_skills':top_soft_skills },
                       ignore_index=True)
    return output
def gen_spec_key_skills(dataframe):

    # Dataframe from title and Description Column
    title_desc_df = dataframe.iloc[:, [1, 3]].copy()

# Cleaning of data Convert all text to lower cases Delete all tabulation,spaces, and new lines, Delete all numericals Delete nltk's defined stop words,Lemmatize text   
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
    title_desc_df ['Description'] = title_desc_df ['Description'].str.replace('[^\w\s]',' ')
    ## digits
    title_desc_df ['Description'] = title_desc_df ['Description'].str.replace('\d+', '')

    #remove stop words
    stop = stopwords.words('english')
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            ## lemmatization
    title_desc_df ['Description'] = title_desc_df ['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
         #--   st.dataframe(title_desc_df)

        # remove further meaningless words PERHAPS CSV OF STOP WORDS
       
    other_stop_words = ['gbp','requirement','required','per','year','junior', 'senior','experience','etc','job','work','company','technique',
                    'candidate','skill','skills','language','menu','inc','new','plus','years',
                   'technology','organization','ceo','cto','account','manager','scientist','mobile',
                    'developer','product','revenue','strong','full','salary']

    title_desc_df['Description'] = title_desc_df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stop_words))
          #--  st.dataframe(title_desc_df)

        # train the naive bayes algorithm.

        ## Converting text to features 
    vectorizer = TfidfVectorizer()
        #Tokenize and build vocabulary
    X = vectorizer.fit_transform(title_desc_df.Description)
    y = title_desc_df.Title

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
        
        # Fit model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
         ## Predict
    y_predicted = clf.predict(X_test)

    #SKILL RETREIEVAL
    technical_skills = retrieve_general_skill_list()
    soft_skills = ['communication', 'teamwork','problem solving', 'time management','critical thinking','decision-making','organizational','stress management','leadership','creative','ambitious',
                   'determined','resourceful','persuasive','public speaking','responsible','hard working']
    
    
    feature_array = vectorizer.get_feature_names()
            # number of overall model features
    features_numbers = len(feature_array)
## max sorted features number
    n_max = int(features_numbers * 0.1)

##initialize output dataframe
    output = pd.DataFrame()
    for i in range(0,len(clf.classes_)):
        print("\n****" ,clf.classes_[i],"****\n")
        class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
        raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    ## Extract technical skills
        top_technical_skills= list(set(technical_skills).intersection(raw_skills))[:5]
    ## Soft Skills
        top_soft_skills = list(set(soft_skills).intersection(raw_skills))[:5]

    # transform list to string
        txt = " ".join(raw_skills)
        blob = TextBlob(txt)
    #top 5 adjective
        top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:5]
    #print("Top 5 : top_technical_skills",top_soft_skills)
    
        output = output.append({'Title':clf.classes_[i],
                        'technical_skills':top_technical_skills,
                        'soft_skills':top_soft_skills },
                       ignore_index=True)
    return output
