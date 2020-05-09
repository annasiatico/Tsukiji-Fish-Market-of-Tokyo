import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prep_data(df):

    data_cleaned = df.reset_index(drop=True)

    log_weight = np.log(data_cleaned['Weight'])

    data_cleaned['log_weight'] = log_weight

    data_cleaned = data_cleaned.drop(['Weight'],axis=1)

    data_no_multicollinearity = data_cleaned.drop(['Length2', 'Length3', 'Width'],axis=1)

    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

    cols = ['log_weight', 'Length1', 'Height', 'Species_Parkki',
       'Species_Perch', 'Species_Pike', 'Species_Roach', 'Species_Smelt',
       'Species_Whitefish']

    data_preprocessed = data_with_dummies[cols]

    targets = data_preprocessed['log_weight']

    inputs = data_preprocessed.drop(['log_weight'],axis=1)

    scaler = StandardScaler()

    scaler.fit(inputs)

    inputs_scaled = scaler.transform(inputs)

    X = inputs_scaled

    y = targets

    return X, y
