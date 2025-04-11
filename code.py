# =======================================================
# Predicting Product Shipment Weight: Code File
# =======================================================
# Author: Reddygari Sri Tejaswini
# Project: Predicting Product Shipment Weight for Reliance Industries Limited
# Date: 14/03/2025
# =======================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load & Understand Data
data = pd.read_csv('shipment_data.csv')
print(data.info())
print(data.describe())

# 3. Data Wrangling
data.fillna(data.median(), inplace=True)
data.drop_duplicates(inplace=True)
data.rename(columns={
    'wh_capacity': 'WarehouseCapacity',
    'prod_wt': 'ShipmentWeight',
    'refill': 'RefillCount',
    'breakdown': 'BreakdownFrequency'
}, inplace=True)
# Outlier removal using IQR
Q1 = data['ShipmentWeight'].quantile(0.25)
Q3 = data['ShipmentWeight'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['ShipmentWeight'] >= Q1 - 1.5 * IQR) & (data['ShipmentWeight'] <= Q3 + 1.5 * IQR)]

# 4. Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 5))
sns.histplot(data['ShipmentWeight'], bins=30, kde=True, color='blue')
plt.title('Histogram of Shipment Weights')
plt.xlabel('Shipment Weight (tons)')
plt.ylabel('Frequency')
plt.savefig("Histogram_ShipmentWeights.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='WarehouseCapacity', y='ShipmentWeight', data=data)
plt.title('Warehouse Capacity vs. Shipment Weight')
plt.xlabel('Warehouse Capacity')
plt.ylabel('Shipment Weight (tons)')
plt.savefig("Scatter_Warehouse_vs_Shipment.png")
plt.close()

plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig("Correlation_Matrix.png")
plt.close()

# 5. Model Development
features = ['WarehouseCapacity', 'RefillCount', 'BreakdownFrequency']
target = 'ShipmentWeight'
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression -- MAE: {:.2f}, RMSE: {:.2f}, R2: {:.2f}".format(
    mean_absolute_error(y_test, y_pred_lr),
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    r2_score(y_test, y_pred_lr)))

# Model 2: Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree -- MAE: {:.2f}, RMSE: {:.2f}, R2: {:.2f}".format(
    mean_absolute_error(y_test, y_pred_dt),
    np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    r2_score(y_test, y_pred_dt)))

# Model 3: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest -- MAE: {:.2f}, RMSE: {:.2f}, R2: {:.2f}".format(
    mean_absolute_error(y_test, y_pred_rf),
    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    r2_score(y_test, y_pred_rf)))
