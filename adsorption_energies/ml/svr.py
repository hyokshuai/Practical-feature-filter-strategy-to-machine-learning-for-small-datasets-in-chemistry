import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load and clean data
path = 'data.xlsx'
data = pd.read_excel(path)
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(training_set, target_set, test_size=0.25, random_state=62)

# Define parameter grid for GridSearchCV
param_grid = {
    'kernel': ['rbf'],
#    'C': np.arange(1,200,1),
    'C': [2],
    'gamma': ['scale'],
#    'epsilon': np.arange(0,0.1,0.1)
    'epsilon': [0.04]
}

# Create and train SVR model with GridSearchCV
model = SVR()
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Save the best model
model_filename = 'best_svr_model.pkl'
joblib.dump(best_model, model_filename)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_best = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²) score:", r2_best)
print("Mean Absolute Error:", mae)
print("Best Parameters:", best_params)

# Save predictions to Excel
prediction_df = pd.DataFrame({
    'True Values': y_test,
    'Predictions': y_pred
})
prediction_df.to_excel('results.xlsx', index=False)

# Plot predictions vs true values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.3, label='True')
plt.plot(range(len(y_pred)), y_pred, color='blue', alpha=0.3, label='Prediction')
plt.ylabel('Sublimation Enthalpy')
plt.xlabel('Index')
plt.legend()
plt.savefig("comparison.png", dpi=300)
plt.show()

