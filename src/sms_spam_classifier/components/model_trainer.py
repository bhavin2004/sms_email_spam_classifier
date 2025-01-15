from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np
from src.sms_spam_classifier.utlis import save_pkl
from src.sms_spam_classifier.utlis import 

##importing all models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ModelTrainerConfig():
    model_path = Path('artifacts/model.pkl')
    
class ModelTrainer():
    
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.similarity_path),exist_ok=True)

    def initiate_model_trainer(self,x_train,x_test,y_train,y_test):
        try:
            logging.info("Initiating the Model Trainer")
            models={
                LogisticRegression(solver='liblinear',penalty='l1'),
                SVC(kernel='sigmoid',gamma=0.1),
                MultinomialNB(),
                BernoulliNB(),
                GaussianNB(),
                DecisionTreeClassifier(max_depth=5),
                ExtraTreeClassifier(max_depth=5),
                ExtraTreesClassifier(n_estimators=50,random_state=2),
                RandomForestClassifier(n_estimators=50,random_state=3),
                AdaBoostClassifier(n_estimators=50,random_state=3),
                GradientBoostingClassifier(n_estimators=50,random_state=3),
                HistGradientBoostingClassifier(n_estimators=50,random_state=3),
                BaggingClassifier(n_estimators=50,random_state=3),
                KNeighborsClassifier()
                
            }
            logging.info("Starting the model evaluation")           
            evaluated_models=evaluate_the_models(models,x_train,x_test,y_train,y_test)
            logging.info("Evaluation of models completed")
            logging.info(evaluated_models)
            # logging.info("\n"+"\n".join([str(i) for i in evaluated_models.items()]))
            logging.info(f"Every model performance=>\n"+"\n".join([str(i) for i in evaluated_models.items()]))
            logging.info("Finding best model using best model function")
            best_model_name,best_model_trained,best_score=best_model(models,evaluated_models)
            print(f'Best Model Name: {best_model_name}')
            print(f'Best Model Score: {best_score}')
            
            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(f"Best Model Score: {best_score}")
            logging.info("Saving the best model")
            save_pkl(obj=best_model_trained,obj_path=self.config.model_file_path)
            logging.info("Model Pickle file is saved")
        except Exception as e:
            logging.error("Error occured during model training: {}".format(e))
            raise CustomException(e,sys)  