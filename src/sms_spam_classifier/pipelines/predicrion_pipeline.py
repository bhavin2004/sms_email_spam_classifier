from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
from src.utlis import load_pkl
import sys
import os
import pandas as pd
from src.movie_recommendation_system.pipelines.training_pipeline import Training_Pipeline
@dataclass
class PredictionPipelineConfig():
    similarity_path = Path('artifacts/similarity.pkl')
    data_path = Path('artifacts/processed_data/data.csv')
    
    
    
class PredictionPipeline():
    def __init__(self):
        self.config = PredictionPipelineConfig()
        
    def run_pipeline(self,movie):
        try:
            logging.info("Starting Prediction Pipeline")
            similarity = load_pkl(self.config.similarity_path)
            data = pd.read_csv(self.config.data_path)
            logging.info("Successfully Loaded neccessary files")
        
            return self.recommend(movie,data,simalarities=similarity)
        except Exception as e:
            logging.error(f"Error Occurred in Prediction Pipeline due to {e}")
            raise CustomException(e,sys)

    def recommend(self,movies,new_df,simalarities):
        movies_index= new_df[new_df['title']==movies].index[0]
        distance = simalarities[movies_index]
        movies_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
        
        recommended_movies_dict ={}
        for i in movies_list:
            movie_id=new_df.iloc[i[0]]['id']
            recommended_movies=new_df.iloc[i[0]].title
            recommended_movies_dict[movie_id]=recommended_movies
            
        return recommended_movies_dict
            
            
if __name__ == "__main__":
    obj = PredictionPipeline()
    print(obj.run_pipeline("Batman"))