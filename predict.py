from joblib import load
from preprocess import prep_data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def predict_from_csv(path_to_csv):

    fish_data = pd.read_csv(path_to_csv)

    X,y = prep_data(fish_data)

    reg = load("reg.joblib")

    predictions = np.exp(reg.predict(X))

    df_preds = pd.DataFrame(predictions, columns=['Prediction'])
    y = y.reset_index(drop=True)
    df_preds['Target'] = np.exp(y)
    pd.options.display.max_rows = 999
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    return df_preds

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)

######
### CHECK MSE ###
from sklearn.metrics import mean_squared_error
ho_predictions = predict_from_csv("fish_holdout_demo.csv")
ho_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
ho_mse = mean_squared_error(ho_truth, ho_predictions["Prediction"])
print(round(ho_mse, 2))
root_mse = np.sqrt(ho_mse)
print(round(root_mse, 2))
######

