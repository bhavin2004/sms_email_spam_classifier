import sys
from src.logger import logging
from src.exception import CustomException
from src.sms_spam_classifier.components.data_ingestion import DataIngestion
from src.sms_spam_classifier.components.data_transformation import DataTransformation
from src.sms_spam_classifier.components.model_trainer import ModelTrainer





class Training_Pipeline():
    def __init__(self):
        self.data_ingestion=DataIngestion()
        self.data_transformation=DataTransformation()
        self.model_trainer=ModelTrainer()
    
    def run_pipeline(self):
        try:
            
            logging.info('Running Training Pipeline')
            train_path,test_path =self.data_ingestion.initiate_data_ingestion()
            print(train_path,test_path)
            data_set=self.data_transformation.initiate_data_transformation(train_path,test_path)
            # print(data_set)
            self.model_trainer.initiate_model_trainer(data_set)
            
        except Exception as e:
            
            logging.error(f"Error occurred in training pipeline due to {e}")
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = Training_Pipeline()
    obj.run_pipeline()