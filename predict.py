### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def predict_from_csv(path_to_csv):

    fish_data = pd.read_csv(path_to_csv)


    X,y = prep_data(fish_data)

    reg = load("reg.joblib")

    predictions = np.exp(reg.predict(X))
    #predictions = reg.predict(X)

    mse = mean_squared_error(y, predictions)
    print(mse)

    return predictions

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)
######

### WE WRITE THIS ###
# from sklearn.metrics import mean_squared_error
# ho_predictions = predict_from_csv("fish_holdout_demo.csv")
# ho_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
# ho_mse = mean_squared_error(ho_truth, ho_predictions)
# print(ho_predictions)
# print(ho_truth)
# print(ho_mse)
######

