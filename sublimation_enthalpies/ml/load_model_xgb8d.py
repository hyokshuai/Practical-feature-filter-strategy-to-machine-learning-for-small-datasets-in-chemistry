import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import shap

# Load data
path = 'data_200.xlsx'
data = pd.read_excel(path)
data_pre = pd.read_excel('prediction.xlsx')

# Clean data
data = data[~data['enthalpy'].isin(['NAN'])]
data = data.dropna(axis=0, how='any')

# Define features and target (For 8D)
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

# Load model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)
print("MAE:", mae)
print("Prediction:", y_pred)
print("True:", y_test)
print("R-squared (R²) score:", r2_best)

# Get feature importances from the trained model
feature_importances = model.feature_importances_

# Sort the feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = np.array(columns)[sorted_indices]
feature_importance_df = pd.DataFrame({
    'Feature': sorted_feature_names,
    'Importance': sorted_feature_importances
})
feature_importance_df.to_excel('feature.xlsx', index=False)

# Create a bar plot to visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# SHAP values calculation and visualization
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, feature_names=columns, show=False)
plt.xlabel("SHAP Value\nImpact on model output", fontsize=12)
plt.xticks(fontsize=15, weight='bold')
plt.yticks(fontsize=10, weight='bold')
plt.xlim(-5, 5)
plt.gcf().set_size_inches(10, 6)
plt.savefig('shap_summary_plot.png', dpi=300)
plt.show()

# Generate and predict new data
atoms = np.arange(2, 3, 1)
r10 = np.arange(100, 160, 70)
r20 = np.arange(130, 1400, 30)
m10 = np.arange(150, 246, 200)
m20 = np.arange(6, 51, 1)
en10 = np.arange(1, 2, 2)
en20 = np.arange(1.5, 4, 5)
tm = np.arange(23, 4000, 300)

# Genrate data for heat plot
pre_data = []
for z in range(len(atoms)):
    for i in range(len(r10)):
        for j in range(len(r20)):
            for k in range(len(m10)):
                for o in range(len(m20)):
                    for p in range(len(en10)):
                        for q in range(len(en20)):
                            for r in range(len(tm)):
                                pre_data.append([atoms[z], r10[i], r20[j], m10[k], m20[o], en10[p], en20[q], tm[r]])

pre_set = pd.DataFrame(pre_data, columns=columns)
X_test1 = scaler.transform(pre_set.values)

# Make predictions for head plot
y_pred1 = model.predict(X_test1)
pre_set['Prediction'] = y_pred1

# Save predictions to Excel
prediction = pd.concat([data_pre1, pd.DataFrame(y_pred, columns=['Prediction'])], axis=1)
prediction.columns = columns + ["Prediction"]
prediction.to_excel('all_prediction.xlsx', index=False)
pre_set.to_excel('heat_map.xlsx', index=False)
