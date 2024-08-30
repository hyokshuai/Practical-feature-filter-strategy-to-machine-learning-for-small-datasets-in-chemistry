import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

X_train, X_test, y_train, y_test = train_test_split(final_data.values, target_set, test_size=0.25, random_state=62)

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
best_params = grid_search.best_params_

# Make predictions on the test set using the best model
y_pred, y_pred_std = best_model.predict(X_test, return_std=True)

# Plot
x = range(len(y_pred))
plt.plot(x, y_test, color='red', alpha=0.3, label='True')
plt.plot(x, y_pred, color='blue', alpha=0.3, label='Prediction')
plt.ylabel('Sublimation Enthalpy')
plt.xlabel('Values')
plt.legend()
plt.savefig("comparison.png", dpi=300)
plt.show()

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²) score:", r2_best)
print("Mean Absolute Error:", mae)
print("Optimized Model - R-squared:", r2_best)
print("Best Parameters:", best_params)

# Save predictions
y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
y_pred_df.to_excel('results.xlsx', index=False)
