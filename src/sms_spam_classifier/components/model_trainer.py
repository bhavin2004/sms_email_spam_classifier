from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np
from src.sms_spam_classifier.utlis import save_pkl, evaluate_the_models, top_n_models_with_tuning

# Importing all models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ModelTrainerConfig:
    model_path = Path("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def initiate_model_trainer(self, x_train, x_test, y_train, y_test, top_n=3):
        try:
            logging.info("Initiating the Model Trainer")
            models = {
                "LogisticRegression": LogisticRegression(solver="liblinear", penalty="l1"),
                "SVC": SVC(kernel="sigmoid", gamma=0.1),
                "MultinomialNB": MultinomialNB(),
                "BernoulliNB": BernoulliNB(),
                "GaussianNB": GaussianNB(),
                "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5),
                "ExtraTreeClassifier": ExtraTreeClassifier(max_depth=5),
                "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=50, random_state=2),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=3),
                "AdaBoostClassifier": AdaBoostClassifier(n_estimators=50, random_state=3),
                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=50, random_state=3),
                # "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
                "BaggingClassifier": BaggingClassifier(n_estimators=50, random_state=3),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }
            logging.info("Starting the model evaluation")
            evaluated_models = evaluate_the_models(models, x_train, x_test, y_train, y_test)
            logging.info("Evaluation of models completed")
            logging.info(f"Every model performance=>\n" + "\n".join([str(i) for i in evaluated_models.items()]))

            # Find the top N models and perform hyperparameter tuning
            logging.info(f"Finding the top {top_n} models and tuning them")
            best_model_name, best_model_trained, best_scores = top_n_models_with_tuning(
                models, evaluated_models, x_train, y_train, x_test, y_test, n=top_n
            )
            logging.info(f"Tuned top {top_n} models successfully")

            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(f"Best Model Scores: {best_scores}")
            logging.info("Saving the best model")
            save_pkl(obj=best_model_trained, obj_path=self.config.model_path)
            logging.info("Model Pickle file is saved")

        except Exception as e:
            logging.error("Error occurred during model training: {}".format(e))
            raise CustomException(e, sys)
