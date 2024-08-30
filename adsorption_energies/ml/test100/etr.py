import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor

       
#get data
path='data.xlsx'
data=pd.read_excel(path)

#data clean
data=data[~data['CH3'].isin(['NAN'])]
data=data.dropna(axis=0, how='any')
#data=data.drop(data[data['enthalpy']>5.5].index)
columns=['G','SE']
#columns=['AN','AM','G','P','R','EN','mp','bp','hfus','density','IE','SE']
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
        'n_estimators': [118],  # Number of trees in the forest
        'max_depth': [6],      # Maximum depth of the tree
        'min_samples_split': [2],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [3]     # Minimum number of samples required to be at a leaf node
    }

    # Create an ExtraTreesRegressor model
    etr_model = ExtraTreesRegressor(random_state=42)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=etr_model, param_grid=param_grid, cv=5)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best parameters and the best score
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # Get the best model
    best_etr_model = grid_search.best_estimator_
    y_pred = best_etr_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2_best = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    # Append (i, X_train) tuple to the data list
    test.append((i, mae, rmse, r2_best))
    
test= pd.DataFrame(test, columns=['Random Seed', 'MAE','RMSE','R2'])

test.to_excel('tests.xlsx')

