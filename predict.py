### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def predict_from_csv(path_to_csv):

    fish_data = pd.read_csv(path_to_csv)


    X,y = prep_data(fish_data)

    reg = load("reg.joblib")

    predictions = np.exp(reg.predict(X))

    df_pf = pd.DataFrame(predictions, columns=['Prediction'])
    y = y.reset_index(drop=True)
    df_pf['Target'] = np.exp(y)
    #df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
    #df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
    pd.options.display.max_rows = 999
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    return df_pf

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)

######
### WE WRITE THIS ###
from sklearn.metrics import mean_squared_error
ho_predictions = predict_from_csv("fish_holdout_demo.csv")
ho_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
ho_mse = mean_squared_error(ho_truth, ho_predictions["Prediction"])
print(ho_mse)
root_mse = np.sqrt(ho_mse)
print(root_mse)
######

