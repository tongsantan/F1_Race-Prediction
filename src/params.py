def params():
    params={
          
              "Hist Gradient Boosting":{
              'learning_rate':[.1,.01,.05,.001],
              'max_depth': [6,8,10]
                  },
      
              "XGBRegressor":{
              'learning_rate':[.1,.01,.05,.001],
              'n_estimators': [8,16,32,64,128,256]
                              },
      
              "CatBoosting Regressor":{
              'depth': [6,8,10],
              'learning_rate': [0.01, 0.05, 0.1],
              'iterations': [30, 50, 100]
                                      },  

      }
    return params