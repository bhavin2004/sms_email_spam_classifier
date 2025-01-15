from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
from src.sms_spam_classifier.utlis import load_pkl,transform_text
import sys
import os
import pandas as pd
import nltk

@dataclass
class PredictionPipelineConfig():
    model_pkl_path = Path('artifacts/model.pkl')
    vectorizer_pkl_path = Path('artifacts/vectorizer.pkl')
    
    
    
class PredictionPipeline():
    def __init__(self):
        self.config = PredictionPipelineConfig()
        
    def run_pipeline(self,sms:str):
        try:
            logging.info("Starting Prediction Pipeline")
            model = load_pkl(self.config.model_pkl_path)
            vectorizer = load_pkl(self.config.vectorizer_pkl_path)
            logging.info("Successfully Loaded neccessary files")
            sms = [transform_text(sms)]
            print(sms)
            sms_vector = vectorizer.transform(sms).toarray()
            print(sms_vector)
            prediction = model.predict(sms_vector)
            return 'Spam' if prediction == 1 else "Ham"
        except Exception as e:
            logging.error(f"Error Occurred in Prediction Pipeline due to {e}")
            raise CustomException(e,sys)

            
            
if __name__ == "__main__":
    obj = PredictionPipeline()
    print(obj.run_pipeline("+123 Congratulations - in this week's competition draw u have won the å£1450 prize to claim just call 09050002311 b4280703. T&Cs/stop SMS 08718727868. Over 18 only 150ppm"))