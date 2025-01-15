from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import os
import sys
# from src.utlis import get_name_data,get_cast_names,fetch_director

@dataclass
class DataTransformationConfig():
    transformed_data_path = Path('artifacts/')
    
class DataTransformation():
    
    def __init__(self):
        self.config = DataTransformationConfig()
        
        
    def initiate_data_transformation(self,data_path):
        try:
            logging.info("Data Transformation is Initiated")
            df = pd.read_csv(data_path,)
    
            ##Columns to have in the Movies dataFrame
            cols = ['genres','id','keywords','title','overview','cast','crew']
            
            df = df[cols]
            
            ##filling nan values with "" blank
            
            df = df.fillna("")
            ##Extracting important tags from genres and keyword cols
            df['genres']= df.genres.apply(get_name_data)
            df['keywords']= df.keywords.apply(get_name_data)
            
            ##Extracting top n cast from cast columns where n = 3
            df['cast']= df.cast.apply(lambda x: get_cast_names(x,n=3))
            
            
            ##Extracting Director's name from the crew column
            df['crew']=df['crew'].apply(fetch_director)
            
            ##Spliting the overview col to make them tag also
            df['overview']=df['overview'].str.split()
            ## Now Creating A new columns by combining cols->keywords,genres,cast,crew
            df['tags']= df['overview']+df['genres']+df['keywords']+df['cast']+df['crew']           
            
            logging.info('Convert all the cols in create tag column')
            ##Just have id,title and tags
            df = df[['id','title','tags']]
            
            ##Loweing the tags for not generating Vocabulary big and for simplification
            df['tags']=df['tags'].apply(lambda x: " ".join(x).lower())
            logging.info(f"The head of the transformed dataset=>\n{df.head().to_string()}") 
            
            logging.info(f"Transformed Data is saved at {self.config.transformed_data_path}")
            df.to_csv(self.config.transformed_data_path)
             
             
            return df     

        except Exception as e:
            logging.error(f"Error occurred in data transformation due to {e}")
            raise CustomException(e,sys)
