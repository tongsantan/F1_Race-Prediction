from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pgbm.sklearn import HistGradientBoostingRegressor

def models():
    models = {
        
            "Hist Gradient Boosting": HistGradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),

        }
    return models