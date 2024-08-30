import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load and clean data
data = pd.read_excel('data_200.xlsx')
data_pre = pd.read_excel('prediction.xlsx')

# Clean data
data = data[~data['enthalpy'].isin(['NAN'])]
data = data.dropna(axis=0, how='any')

# Define features and target
# 6D input
#columns=["number_atoms","R_1","R_2","en_1","en_2","Tm"]
# 7D input
#columns=["R_1","R_2","m_1","m_2","en_1","en_2","Tm"]
# 8D input
columns = ["number_atoms", "R_1", "R_2", "m_1", "m_2", "en_1", "en_2", "Tm"]
final_data = data[columns]
target_set = np.asarray(data['enthalpy']).astype(np.float32)

# Standardization
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(final_data.values)

# Define the prediction dataset
data_pre1 = data_pre[columns]
X_test = scaler.transform(data_pre1.values)
y_test = np.asarray(data_pre['enthalpy']).astype(np.float32)

# XGBoost model and parameter grid
params = {'objective': 'reg:squarederror', 'eval_metric': 'mae'}
model = xgb.XGBRegressor(**params)
param_grid = {
#    'learning_rate': np.arange(0.01,1,0.01),
#    'max_depth': range(1,200,2),
#    'n_estimators': range(1,150,1)    
    'learning_rate': [0.4],
    'max_depth': [10],
    'n_estimators': [17]
}
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'R2': 'r2'
}

# Grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring=scoring, refit='R2')
grid_search.fit(X_train, target_set)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Save the model
best_model.save_model('xgb_model.json')

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)

# Print evaluation metrics
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
prediction_df = pd.DataFrame({
    'True': y_test,
    'Prediction': y_pred
})
prediction_df.to_excel('results.xlsx', index=False)

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
