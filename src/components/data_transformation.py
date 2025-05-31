import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_columns = ['type','yearpublished','minplayers','maxplayers','playingtime','minage','users_rated',
                                'bayes_average_rating','total_traders','total_wanters','total_wishers','average_weight']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info('Initiating Data Transformation')

            train_arr = pd.read_csv(train_data_path)
            test_arr = pd.read_csv(test_data_path)

            logging.info('Read the training and testing data')

            target = 'average_rating'

            train_input_feature = train_arr.drop(target,axis=1)
            train_target_feature = train_arr[target]
            test_input_feature = test_arr.drop(target,axis=1)
            test_target_feature = test_arr[target]

            preprocessor_obj = self.get_data_transformation_obj()
            logging.info('Loaded the preprocessor object')

            transformed_train_input_feature = preprocessor_obj.fit_transform(train_input_feature)
            transformed_test_input_feature = preprocessor_obj.transform(test_input_feature)

            transformed_train_arr = np.c_[transformed_train_input_feature,np.array(train_target_feature)]
            transformed_test_arr = np.c_[transformed_test_input_feature,np.array(test_target_feature)]

            logging.info('Transformed the training and testing data')

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_path,obj=preprocessor_obj)

            logging.info('Saved the preprocessor object')

            return(
                transformed_train_arr,
                transformed_test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)