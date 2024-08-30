# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:13:33 2023

@author: Hasse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
target_set = np.asarray(data.iloc[:, 13]).astype(np.float32)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
training_set = scaler.fit_transform(final_data.values)

# Split data
X_train, X_test, y_train, y_test = train_test_split(final_data.values, target_set, 
                                                    test_size=0.25, random_state=22)

# Build and tune the model
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

param_grid = {
    #    'learning_rate': np.arange(0.1,1,0.1),
    #    'max_depth': range(1,20,2),
    #    'n_estimators': range(1,20,1),
    'learning_rate': [0.2],
    'max_depth': [2],
    'n_estimators': [9]
}

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'R2': 'r2'
}

grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring=scoring, refit='R2')
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Save the model
best_model.save_model('xgb_model.json')

# Make predictions
y_pred = best_model.predict(X_test)

# Print results
print("Prediction:", y_pred)
print("True:", y_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²) score:", r_squared)
print("Mean Absolute Error:", mae)
print("Optimized Model - R-squared:", r_squared)
print("Best Parameters:", best_params)

# Save results to Excel
results_df = pd.DataFrame({'True': y_test, 'Prediction': y_pred})
results_df.to_excel('results.xlsx', index=False)

# Plot results
plt.figure(figsize=(10, 6))
x = range(len(y_pred))
plt.plot(x, y_test, color='red', alpha=0.3, label='True')
plt.plot(x, y_pred, color='blue', alpha=0.3, label='Prediction')
plt.ylabel('Sublimation Enthalpy')
plt.xlabel('Values')
plt.legend()
plt.savefig("comparison.png", dpi=300)
plt.show()

