import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from myproject.path_utils import data_path
# Load the dataset
train_csv = data_path("Q1", "house_price.csv")
data = pd.read_csv(train_csv)
print(train_csv)
# Define features and target
X = data[['bedroom', 'size']]
y = data['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- LinearRegression -----
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Metrics for LinearRegression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

# # ----- SGDRegressor -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000)
sgd.fit(X_train_scaled, y_train)
y_pred_sgd  = sgd.predict(X_test_scaled)

# # Metrics for SGDRegressor
mae_sgd = mean_absolute_error(y_test, y_pred_sgd )
mse_sgd = mean_squared_error(y_test, y_pred_sgd )
rmse_sgd = np.sqrt(mse_sgd)
mape_sgd = mean_absolute_percentage_error(y_test,y_pred_sgd)

# Display results
print("=== LinearRegression Results ===")
print("Coefficients:")
print(pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient']), "\n")
print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"MAPE: {mape_lr:.2%}\n")

print("=== SGDRegressor Results ===")
print("Coefficients:")
print(pd.DataFrame(sgd.coef_, X.columns, columns=['Coefficient']), "\n")
print(f"MAE: {mae_sgd:.2f}")
print(f"MSE: {mse_sgd:.2f}")
print(f"RMSE: {rmse_sgd:.2f}")
print(f"MAPE: {mape_sgd:.2%}")


# MAE gives your “typical” dollar error (~$72 k).
# MSE squares large misses (you get 8.6 billion in squared‑dollar units).
# RMSE brings it back to dollars (~$92 k), still punishing big errors more.
# MAE - want a straightforward “average error” that won’t be skewed by a couple of bad predictions.
# RMSE - preventing large misses is crucialsss
# RMSE for LinearRegression: 92792.37
# RMSE for SGDRegressor: 92688.92
