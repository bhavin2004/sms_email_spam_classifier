import requests
from src.exception import CustomException
import sys
from ast import literal_eval
from nltk.data import find
import nltk
import pickle
import os
import logging

logging.basicConfig(
    filename="logs/nltk.log",
    level=logging.INFO
)





def download_nltk_resource(resource_name):
    try:
        find(resource_name)
    except LookupError:
        nltk.download(resource_name)

# Example




def get_name_data(obj):
    names = [i['name'].replace(' ','') for i in literal_eval(obj)]
    return names


def get_cast_names(obj,n):
    tags = []
    count = 0
    for i in literal_eval(obj):
        if count!=n:
            tags.append(i['name'].replace(' ',''))
            count+=1
        else:
            break
    return tags

def fetch_director(obj):
    director_name = []
    for i in literal_eval(obj):
        if i['job']=='Director':
            director_name.append(i['name'].replace(' ',''))
            break
    return director_name



def save_pkl(obj,obj_path):
    try:
        folder_dir=os.path.dirname(obj_path)
        os.makedirs(folder_dir,exist_ok=True)
        
        with open(obj_path,'wb') as f:
            pickle.dump(obj,f)
    except Exception as e:
        
            raise CustomException(e,sys)
 


def fetch_poster(movie_id:int):
    
    responese = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=332600740a4d938b7c5f33048df5b6d6&language=en-US')
    
    data = responese.json()
    
    return "https://image.tmdb.org/t/p/w500"+data['poster_path']    

def load_pkl(pkl_path):
    try:
        with open(pkl_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)
logging.info(download_nltk_resource('omw-1.4'))  # For extended WordNet data
logging.info(download_nltk_resource('wordnet'))
logging.info(download_nltk_resource('punkt'))    # For tokenization
logging.info(download_nltk_resource('punkt_tab'))



# import requests

# url = "https://api.themoviedb.org/3/authentication"

# headers = {
#     "accept": "application/json",
#     "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzMzI2MDA3NDBhNGQ5MzhiN2M1ZjMzMDQ4ZGY1YjZkNiIsIm5iZiI6MTczNjc5MTMxMi4zNTUsInN1YiI6IjY3ODU1NTEwNGJmZDdlZjU1ZGJiMzhkMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.kYAi5NXaKuwf5bPSJbjNupZhnmr4ZyxKpEXC_RnZEqk"
# }

# response = requests.get(url, headers=headers)

# print(response.text)
# fetch_poster(550)