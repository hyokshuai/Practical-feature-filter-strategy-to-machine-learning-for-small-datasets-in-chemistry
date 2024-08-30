import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Get data
path = 'data.xlsx'
data = pd.read_excel(path)

# Data cleaning
data = data[~data['CH3'].isin(['NAN'])]
data = data.dropna(axis=0, how='any')
columns = ['G', 'SE']
final_data = data[columns]

# Model building
ch3 = data.iloc[:, 13]
target_set = np.asarray(ch3).astype(np.float32)
training_set = final_data.values

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(training_set)
training_set = scaler.transform(training_set)

# Initialize an empty list to store test results
test_results = []

# Loop over 100 different random seeds
for i in range(1, 101):
    # Split the data using the current random seed i
    X_train, X_test, y_train, y_test = train_test_split(final_data.values, target_set, 
                                                        test_size=0.25, random_state=i)
    
    # Define the parameter grid to search over
    length_scales = [2.5]
    kernels = [RBF(length_scale=length_scale) for length_scale in length_scales]
    param_grid = {
        'kernel': kernels,
        'alpha': [0.05]
    }

    # Create the Gaussian process regression model
    model = GaussianProcessRegressor()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred, y_pred_std = best_model.predict(X_test, return_std=True)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2_best = r2_score(y_test, y_pred)

    # Append results to the test_results list
    test_results.append((i, mae, rmse, r2_best))

# Convert test_results to a DataFrame
test_results_df = pd.DataFrame(test_results, columns=['Random Seed', 'MAE', 'RMSE', 'R2'])

# Save the results to an Excel file
test_results_df.to_excel('tests.xlsx', index=False)
