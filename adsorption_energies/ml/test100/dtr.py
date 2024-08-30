import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
    #    'max_depth': range(1,10,1),
    #    'min_samples_split': range(2,10,1),
    #    'min_samples_leaf': range(1,10,1),
    #    'criterion':["squared_error", "friedman_mse", "absolute_error"],
        'max_depth': [4],
        'min_samples_split': [4],
        'min_samples_leaf': [4],
    }

    # Create the Decision Tree Regression model
    regressor = DecisionTreeRegressor()

    # Define a dictionary of scoring metrics you want to use
    scoring = {
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'r2': 'r2',
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring=scoring, refit='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_

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
