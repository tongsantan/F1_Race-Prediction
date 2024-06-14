import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("output","model.pkl")
            preprocessor_path=os.path.join('output','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        grid: int,
        position: int,
        laps: int,
        fastestLap: int,
        rank: float,
        fastestLapSpeed: float,
        drivername, 
        year: int,
        round: int,
        location,
        constructorname,
        status,
        race_age: int):

        self.grid = grid

        self.position = position

        self.laps = laps

        self.fastestLap = fastestLap

        self.rank = rank

        self.fastestLapSpeed = fastestLapSpeed

        self.drivername = drivername

        self.year = year

        self.round = round

        self.location = location

        self.constructorname = constructorname

        self.status = status

        self.race_age = race_age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "grid": [self.grid],
                "position": [self.position],
                "laps": [self.laps],
                "fastestLap": [self.fastestLap],
                "rank": [self.rank],
                "fastestLapSpeed": [self.fastestLapSpeed],
                "drivername": [self.drivername],
                "year": [self.year],
                "round": [self.round],
                "location": [self.location],
                "constructorname": [self.constructorname],
                "status": [self.status],
                "race_age": [self.race_age],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

