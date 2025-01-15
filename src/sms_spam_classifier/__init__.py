import numpy as np
import pickle
import os, sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score
from time import time


def save_pkl(obj, obj_path):
    try:
        folder_dir = os.path.dirname(obj_path)
        os.makedirs(folder_dir, exist_ok=True)
        logging.info(f"Initiating the Saving of Pickle file {obj_path}")
        with open(obj_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.error(
            "Error occured in saving the {} pkl file: {}".format(obj_path, str(e))
        )
        raise CustomException(e, sys)


def evaluate_the_models(models: dict, x_train, x_test, y_train, y_test):
    try:
        model_score = dict()
        for model_name, model in models.items():
            now = time()
            model.fit(x_train, y_train)
            logging.info(
                "Time Taken by model {} to fit the data: {}".format(
                    model_name, time() - now
                )
            )
            y_pred = model.predict(x_test)
            score = calculate_model_score(y_test, y_pred)
            model_score[model_name] = score
        return model_score
    except Exception as e:
        logging.error("Error occured during evaluation all models: {}".format(e))
        raise CustomException(e, sys)


def calculate_model_score(true, predicated):
    # Calculate the accuracy score
    accuracy = accuracy_score(true, predicated)
    # Calculate the Precison
    precision = precision_score(true, predicated)
    return {"accuracy": accuracy, "precision": precision}


def best_model(models: dict, evaluated_models: dict):
    try:
        score_type = "precison"
        max_score = 0
        for i in range(len(list(models.values()))):
            model_scores = list(evaluated_models.values())[i]
            score = model_scores[score_type]
            if score > max_score:
                best_model_name = list(evaluated_models.keys())[i]
                max_score = score
                best_model = list(models.values())[i]
        return best_model_name, best_model, max_score
    except Exception as e:
        logging.error(
            "Error occured during calculating the best model score: {}".format(e)
        )
        raise CustomException(e, sys)


def load_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logging.error("Error occured during loading the pkl file: {}".format(e))
        raise CustomException(e, sys)
