import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
from preprocess import prep_data


df = pd.read_csv("fish_participant.csv")

X,y = prep_data(df)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(x_train,y_train)

dump(reg, "reg.joblib")

#Test the predictions
y_hat = reg.predict(x_train)
y_hat_test = reg.predict(x_test)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)
print (df_pf)

#Check the MSE
mse = round(mean_squared_error(y_train, y_hat),2)
print("MSE: ", mse)
root_mse = round(np.sqrt(mse), 2)
print("RMSE: ", root_mse)