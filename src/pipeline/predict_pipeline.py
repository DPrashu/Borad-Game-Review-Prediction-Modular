import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class CustomData:
    def __init__(self,type,yearpublished,minplayers,maxplayers,playingtime,minage,users_rated,
                bayes_average_rating,total_traders,total_wanters,total_wishers,average_weight):

        self.type = type
        self.yearpublished = yearpublished
        self.minplayers = minplayers
        self.maxplayers = maxplayers
        self.playingtime = playingtime
        self.minage = minage
        self.users_rated = users_rated
        self.bayes_average_rating = bayes_average_rating
        self.total_traders = total_traders
        self.total_wanters = total_wanters
        self.total_wishers = total_wishers
        self.average_weight = average_weight


    def get_data_as_DataFrame(self):
        try:
            dict = {
                'type' : [self.type],
                'yearpublished' : [self.yearpublished],
                'minplayers' : [self.minplayers],
                'maxplayers' : [self.maxplayers],
                'playingtime' : [self.playingtime],
                'minage' : [self.minage],
                'users_rated' : [self.users_rated],
                'bayes_average_rating' : [self.bayes_average_rating],
                'total_traders' : [self.total_traders],
                'total_wanters' : [self.total_wanters],
                'total_wishers' : [self.total_wishers],
                'average_weight' : [self.average_weight]
            }

            return pd.DataFrame(dict)
            
        except Exception as e:
            raise CustomException(e,sys)

class predict_pipe:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model = load_object(model_path)
            preprocessor_obj = load_object(preprocessor_path)

            transformed_data = preprocessor_obj.transform(features)
            result = model.predict(transformed_data)

            return result
            
        except Exception as e:
            raise CustomException(e,sys)