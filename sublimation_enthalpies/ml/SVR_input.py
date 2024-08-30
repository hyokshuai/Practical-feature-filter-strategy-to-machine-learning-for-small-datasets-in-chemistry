import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load and clean data
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

# Standardization
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(final_data.values)

# Define the prediction dataset
data_pre1 = data_pre[columns]
X_test = scaler.transform(data_pre1.values)
y_test = np.asarray(data_pre['enthalpy']).astype(np.float32)

# Define the parameter grid for SVR
param_grid = {
    'kernel': ['rbf'],
#    'C': np.arange(1,200,1),
    'C': [35],
    'gamma': ['scale'],
#    'epsilon': np.arange(0,0.1,0.1)
    'epsilon': [0.04]
}

# Create and train the SVR model
model = SVR()
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, target_set)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Save the model
model_filename = 'best_svr_model.pkl'
joblib.dump(best_model, model_filename)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Prediction:", y_pred)
print("True:", y_test)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²) score:", r_squared)
print("Mean Absolute Error:", mae)
print("Optimized Model - R-squared:", r_squared)
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
plt.ylabel('Sublimation Enthalpy')
plt.xlabel('Values')
plt.legend()
plt.savefig("comparison.png", dpi=300)
plt.show()

# Compute SHAP values
explainer = shap.KernelExplainer(best_model.predict, X_train)
shap_values = explainer.shap_values(X_train)

# Plot SHAP values
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, feature_names=columns, show=False)
plt.xlabel("SHAP Value\nImpact on model output", fontsize=12)
plt.xlim(-5, 5)  # Customize x-axis range
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')
plt.savefig('svr_shap.png', dpi=300)
plt.show()
