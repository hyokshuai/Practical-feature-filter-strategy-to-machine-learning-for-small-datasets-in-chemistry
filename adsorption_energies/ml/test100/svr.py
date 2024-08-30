# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:13:33 2023

@author: Hasse
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#get data
path='data.xlsx'
data=pd.read_excel(path)


#data clean
data=data[~data['CH3'].isin(['NAN'])]
data=data.dropna(axis=0, how='any')
columns=['G','SE']
final_data=data[columns]


#model building
ch3=data.iloc[:,13]
target_set=np.asarray(ch3).astype(np.float32)
training_set=final_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(training_set)
training_set= scaler.transform(training_set)

test=[]

for i in range(1, 101):
    # Split the data using the current random seed i
    X_train, X_test, y_train, y_test = train_test_split(final_data.values, target_set, 
                                                    test_size=0.25, random_state=i)
    param_grid = {
        'kernel': ['rbf'],
    #    'C': np.arange(1,200,1),
        'C': [2],
        'gamma': ['scale'],
    #    'epsilon': np.arange(0,0.1,0.1)
        'epsilon': [0.04]
    }

    # Create the SVR model
    model = SVR()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2_best = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    # Append (i, X_train) tuple to the data list
    test.append((i, mae, rmse, r2_best))
    
test= pd.DataFrame(test, columns=['Random Seed', 'MAE','RMSE','R2'])

test.to_excel('tests.xlsx')

