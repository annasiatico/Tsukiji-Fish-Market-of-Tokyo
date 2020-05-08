import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prep_data(df):

    q = df["Weight"].quantile(0.99)
    data_1 = df[df["Weight"]<q]

    q2 = df['Length1'].quantile(0.99)
    data_2 = data_1[data_1['Length1']<q2]

    q3 = df['Length2'].quantile(0.99)
    data_3 = data_2[data_2['Length2']<q3]

    data_cleaned = data_3.reset_index(drop=True)

    # data_cleaned = df.reset_index(drop=True)

    log_weight = np.log(data_cleaned['Weight'])

    data_cleaned['log_weight'] = log_weight
    #print(data_cleaned)

    data_cleaned = data_cleaned.drop(['Weight'],axis=1)

    data_no_multicollinearity = data_cleaned.drop(['Length1', 'Length2'],axis=1)

    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

    #print (data_with_dummies)

    cols = ['log_weight', 'Length3', 'Height', 'Width', 'Species_Parkki',
        'Species_Perch', 'Species_Pike', 'Species_Roach', 'Species_Smelt',
        'Species_Whitefish']

    data_preprocessed = data_with_dummies[cols]

    #print(data_preprocessed)

    targets = data_preprocessed['log_weight']

    inputs = data_preprocessed.drop(['log_weight'],axis=1)

    scaler = StandardScaler()

    scaler.fit(inputs)

    inputs_scaled = scaler.transform(inputs)

    X = inputs_scaled

    y = targets

    return X, y


#x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)
# y_hat = reg.predict(x_train)
# # print(y_hat)

# y_hat_test = reg.predict(x_test)
# # print(y_hat_test)

# df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
# y_test = y_test.reset_index(drop=True)
# df_pf['Target'] = np.exp(y_test)

# print (df_pf)

# mse = mean_squared_error(y_train, y_hat)
# print(mse)

# root_mse = np.sqrt(mse)
# print(root_mse)



# print(data_cleaned.describe(include='all'))