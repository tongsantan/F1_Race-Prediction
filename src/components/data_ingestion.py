import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    dataset_path: str=os.path.join('data', "F1_race.csv")
    processed_data_path: str=os.path.join('data', "processed_F1_race.csv")
    train_data_path: str=os.path.join('output',"train.csv")
    test_data_path: str=os.path.join('output',"test.csv")
    raw_data_path: str=os.path.join('output',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initiate data ingestion method or component")
        try:
            # 1st Dataset "constructors_mod"
            df_constructor = pd.read_csv('./notebook/raw_data/constructors.csv')
            df_constructor.drop(columns = ['url', 'constructorRef', 'nationality'], axis=1, inplace=True)
            
            # 2nd Dataset "results_mod"
            df_result = pd.read_csv('./notebook/raw_data/results.csv')

            # 3rd Dataset "drivers_mod"

            df_driver = pd.read_csv('./notebook/raw_data/drivers.csv')

            # Extract year in unstructured datetime data 'dob'  

            df_driver['yob'] = pd.to_datetime(df_driver['dob']).dt.year

            # Apply mathematical Calculations to Features such as addition of 'forename' and 'surname' with an underscore

            df_driver['drivername'] = df_driver['forename'] + '_' + df_driver['surname']

            # Drop the column 'url', 'forename', 'surname', 'nationality', 'dob', 'driverRef' which are no longer useful

            df_driver.drop(columns = ['url', 'code', 'forename', 'surname', 'nationality', 'dob', 'driverRef'], axis=1, inplace=True)

            # 4th Dataset "races_mod"

            df_circuit = pd.read_csv('./notebook/raw_data/races.csv')

            # Extract year in unstructured data 'url text'

            df_circuit['year'] = df_circuit['url'].str.extract('(\d+)')

            # Replace NaN value with 2021 in the year column

            df_circuit['year'] = df_circuit['year'].replace(np.nan, 2021)

            df_circuit['year'] = df_circuit['year'].replace(['70'], '2020')

            # convert the year column datatype to integer

            df_circuit['year'] = df_circuit['year'].astype(int)

            # Drop the column 'url', 'time', 'date' which are no longer useful

            df_circuit.drop(columns = ['url', 'time', 'date'], axis=1, inplace=True)

            # 5th Dataset "status_mod"

            df_status = pd.read_csv('./notebook/raw_data/status.csv')

            # joining df_result with df_driver by column driverID, using inner join

            df_result1 = pd.merge(df_result, df_driver, on='driverId')

            # joining df_result1 with df_circuit by column raceID, using inner join

            df_result2 = pd.merge(df_result1, df_circuit, on='raceId')

            # joining df_result2 with df_constructor by column constructorID, using inner join

            df_result3 = pd.merge(df_result2, df_constructor, on='constructorId')

            # joining df_result3 with df_status by column statusID, using inner join

            df_race_finished = pd.merge(df_result3, df_status, on='statusId')

            # Drop the columns 'resultId', 'raceId', 'driverId', 'constructorId', 'number_x', 'positionText', 'positionOrder', 'time', 'milliseconds', 'fastestLapTime', 'statusId', 'number_y' and 'circuitId' which are not useful

            df_race_finished.drop(columns = ['resultId', 'raceId', 'driverId', 'constructorId', 'number_x', 'positionText', 'positionOrder', 'time', 'milliseconds', 'fastestLapTime', 'statusId', 'number_y', 'circuitId'], axis=1, inplace=True)

            # rename the columns accordingly
            df_race_finished.rename(columns={'name_x': 'location', 'name_y': 'constructorname'}, inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.dataset_path),exist_ok=True)

            df_race_finished.to_csv(self.ingestion_config.dataset_path,index=False,header=True)

            return(
                self.ingestion_config.dataset_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)   


    def data_preprocessing_feature_engineering(self):
        logger.info("Data preprocessing and feature engineering") 

        try:
            df_race_finished = pd.read_csv('./data/F1_race.csv')

            # Applying Mathematical substrations between features 'year' and 'yob' to derive the race_age of the driver

            df_race_finished['race_age'] = df_race_finished['year'] - df_race_finished['yob']
            df_race_finished.drop(columns = ['yob'], axis=1, inplace=True)

            df_race_finished = df_race_finished.drop(columns = ['year', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', \
                                                    'quali_date', 'quali_time', 'sprint_date', 'sprint_time'])
            
            # replace the '\N' values with NaN values

            df_race_finished.replace('\\N', np.NaN, inplace = True)

            # convert the columns with the correct datatype 

            df_race_finished['position'] = df_race_finished['position'].astype("Int64")
            df_race_finished['fastestLap'] = df_race_finished['fastestLap'].astype("Int64")
            df_race_finished['rank'] = df_race_finished['rank'].astype("Int64")
            df_race_finished['fastestLapSpeed'] = df_race_finished['fastestLapSpeed'].astype(float)

            os.makedirs(os.path.dirname(self.ingestion_config.processed_data_path),exist_ok=True)

            df_race_finished.to_csv(self.ingestion_config.processed_data_path,index=False,header=True)

            return(
                self.ingestion_config.processed_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)      

    def complete_data_ingestion(self):
        logger.info("Resume data ingestion method or component") 

        try:

            df=pd.read_csv('./data/processed_F1_race.csv')
            logger.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logger.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logger.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)



