import sys
from dataclasses import dataclass
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pgbm.sklearn import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src import logger
from src.utils.common import save_object,evaluate_models
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from src.entity.config_entity import ModelTrainerConfig

## 5. Update the components

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config

    def initiate_model_trainer(self):
        try:
            train_array=np.load(self.config.train_array_path)
            test_array=np.load(self.config.test_array_path)
            logger.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                
                    "Hist Gradient Boosting": HistGradientBoostingRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                }
            params={
                    
                    "Hist Gradient Boosting":{
                        'learning_rate': [0.1], 
                        'max_depth': [10]
                    },
        
                    "XGBRegressor":{
                        'learning_rate': [0.1], 
                        'n_estimators': [256]
                    },
                    "CatBoosting Regressor":{
                        'depth': [8], 
                        'iterations': [100], 
                        'learning_rate': [0.1]
                    },

                }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logger.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logger.info(f"best model: {best_model_name}; best R-squared score: {r2_square}")
            
            return best_model_name, r2_square
            
        except Exception as e:
            raise CustomException(e,sys)