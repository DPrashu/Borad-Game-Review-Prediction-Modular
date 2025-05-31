import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Staring Data Ingestion')

            df = pd.read_csv('notebook\data\games.csv')
            logging.info('Read the data from source')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train Test Spit Initiated')
            train_arr,test_arr = train_test_split(df,test_size=0.2,random_state=64)

            train_arr.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_arr.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Completed')

            return{
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj1 = DataIngestion()
    train_data_path,test_data_path = obj1.initiate_data_ingestion()

    obj2 = DataTransformation()
    train_arr,test_arr,_ = obj2.initiate_data_transformation(train_data_path,test_data_path)

    obj3 = ModelTraining()
    obj3.initiate_model_training(train_arr,test_arr)
