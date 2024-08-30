import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

# Define the parameter grid to search
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
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Get the best model
best_etr_model = grid_search.best_estimator_
y_pred = best_etr_model.predict(X_test)

print("Prediction:", y_pred)
print("True:", y_test)

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

# Save predictions
y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
y_pred_df.to_excel('results.xlsx', index=False)



