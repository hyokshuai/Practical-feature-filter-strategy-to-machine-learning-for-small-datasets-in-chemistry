import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load data
path = 'data_200.xlsx'
data = pd.read_excel(path)
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

# Standardize data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(final_data.values)

# Define the prediction dataset
data_pre1 = data_pre[columns]
X_test = scaler.transform(data_pre1.values)
y_test = np.asarray(data_pre['enthalpy']).astype(np.float32)

# Define the parameter grid for Gaussian Process Regression
# Define length_scale here
length_scales = [3.5]
kernels = [C(1.0) * RBF(length_scale=length_scale) for length_scale in length_scales]

param_grid = {
    'kernel': kernels,  
#    'alpha': np.arange(0.1,1,0.1)
    'alpha': [0.3]
}

# Create and train the Gaussian Process Regression model
model = GaussianProcessRegressor()
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, target_set)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test set using the best model
y_pred, y_pred_std = best_model.predict(X_test, return_std=True)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)

# Print results
print("Prediction:", y_pred)
print("True:", y_test)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2_best}")
print("Best Parameters:", best_params)

# Save predictions to Excel
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
plt.xlabel('Data Number')
plt.ylabel('Sublimation Enthalpy')
plt.legend()
plt.savefig("comparison.png", dpi=300)
plt.show()
