import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_models
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

@dataclass
class ModelTrainingConfig:
    model_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting the datset into dependent and independent features')
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Linear Regressor' : LinearRegression()
            }

            logging.info('Evaluating each and every model')
            report = evaluate_models(x_train,y_train,x_test,y_test,models)

            best_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            logging.info("The best score is {} and the best model is {}".format(best_score,best_model_name))

            best_model = models[best_model_name]

            save_obj(file_path=self.model_training_config.model_path,obj=best_model)

            logging.info('Saved the model')

        except Exception as e:
            raise CustomException(e,sys)

