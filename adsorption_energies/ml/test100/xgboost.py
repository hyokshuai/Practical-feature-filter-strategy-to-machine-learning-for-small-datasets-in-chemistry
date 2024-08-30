import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Load and clean data
path = 'data.xlsx'
data = pd.read_excel(path)

# Data cleaning
data = data[~data['CH3'].isin(['NAN'])]
data = data.dropna(axis=0, how='any')

# Define features and target
columns = ['G', 'SE']
final_data = data[columns]
ch3 = data.iloc[:, 13]
target_set = np.asarray(ch3).astype(np.float32)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
training_set = scaler.fit_transform(final_data.values)

# List to store test results
test_results = []

# Grid search across different random seeds
for i in range(1, 101):
    # Split the data using the current random seed
    X_train, X_test, y_train, y_test = train_test_split(final_data.values, target_set, 
                                                        test_size=0.25, random_state=i)

    # Define the model and parameter grid
    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    param_grid = {
        #    'learning_rate': np.arange(0.1,1,0.1),
        #    'max_depth': range(1,200,2),
        #    'n_estimators': range(30,60,1),
        'learning_rate': [0.2],
        'max_depth': [2],
        'n_estimators': [9]
    }
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'R2': 'r2'
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring=scoring, refit='R2')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions and calculate metrics
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_best = r2_score(y_test, y_pred)

    # Append results for the current random seed
    test_results.append((i, mae, rmse, r2_best))

# Save test results to Excel
test_df = pd.DataFrame(test_results, columns=['Random Seed', 'MAE', 'RMSE', 'R2'])
test_df.to_excel('tests.xlsx', index=False)
