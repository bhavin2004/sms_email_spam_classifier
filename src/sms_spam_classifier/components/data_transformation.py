from src.sms_spam_classifier.utlis import save_pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import os
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
import nltk

import sys
from sklearn.preprocessing import LabelEncoder
# from src.utlis import get_name_data,get_cast_names,fetch_director

# nltk.download('punkt')
# nltk.download('stopwords')

@dataclass
class DataTransformationConfig():
    TfidfVectorizer_data_path = Path('artifacts/vectorizer.pkl')
    
class DataTransformation():
    
    def __init__(self):
        self.config = DataTransformationConfig()
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation is Initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
    
            #deleting the none wanted nan cols
            test_df.dropna(axis=1,inplace=True)
            test_df.dropna(axis=1,inplace=True)
            
            # Renaming the cols
            train_df.rename(columns={'v1':'target','v2':'sms'},inplace=True)
            test_df.rename(columns={'v1':'target','v2':'sms'},inplace=True)

            #encoding the target col where ham=0 and spam=1
            encoder = LabelEncoder()
            train_df.target = encoder.fit_transform(train_df.target)
            test_df.target = encoder.transform(test_df.target)
            train_target = train_df['target']
            test_target = test_df['target']
             
            ## here we will drop duplicates bcoz we know that the duplicated value will affect the text classification by increasing its frequency in dataset
            train_df.drop_duplicates(keep='first',inplace=True)
            test_df.drop_duplicates(keep='first',inplace=True)
            
            #Transforming the data
            train_df['transformed_txt'] = train_df['sms'].apply(self.transform_text)
            test_df['transformed_txt'] = test_df['sms'].apply(self.transform_text)

            #converting txt in vectors
            tf_vectorizer = TfidfVectorizer(max_features=3000)
            x_train = tf_vectorizer.fit_transform(train_df).toarray()
            x_test = tf_vectorizer.transform(test_df).toarray()
            
            logging.info('Created vectors of the sms')
            
            save_pkl(obj=tf_vectorizer,obj_path=self.config.TfidfVectorizer_data_path)

            logging.info('Successfully saved Vectorizer pkl')
            logging.info('Data Transformation is Complete')
            return(
                x_train,
                x_test,
                train_target,
                test_target
            )     

        except Exception as e:
            logging.error(f"Error occurred in data transformation due to {e}")
            raise CustomException(e,sys)
        
    def transform_text(self,text:str,ps=PorterStemmer()):
        text = text.lower()
        text = nltk.word_tokenize(text)
        res=[]
        for i in text:
            if i.isalpha():
                res.append(i)
        text = res[:]
        res.clear()
        for i in text:
            if i not in stopwords.words("english") and i not in punctuation:
                res.append(ps.stem(i))
        return " ".join(res)
