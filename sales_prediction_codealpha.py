
# SALES PREDICTION USING MACHINE LEARNING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# LOAD DATASET

data = pd.read_csv("Advertising.csv")

print("\nDataset Shape:", data.shape)

print("\nFirst Five Rows")
print(data.head())

#DATA CLEANING

data.columns = data.columns.str.strip()

# Remove unnecessary index column

if "Unnamed: 0" in data.columns:
    data.drop(
        "Unnamed: 0",
        axis=1,
        inplace=True
    )

print("\nMissing Values")
print(data.isnull().sum())

data.drop_duplicates(inplace=True)

print(
    "\nDataset Shape After Cleaning:",
    data.shape
)

#EXPLORATORY DATA ANALYSIS

print("\nStatistical Summary")
print(data.describe())

# FEATURE SELECTION

X = data[
    [
        "TV",
        "Radio",
        "Newspaper"
    ]
]

y = data["Sales"]

print("\nSelected Features")
print(X.columns.tolist())

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# DATA TRANSFORMATION

pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

# MODEL TRAINING

print("\nTraining Model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training Completed.")

# SALES PREDICTION

predictions = pipeline.predict(
    X_test
)

# MODEL EVALUATION

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")

#  ACTUAL VS PREDICTED SALES


comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nSample Predictions")
print(
    comparison.head(10)
)


# VISUALIZATION

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
    "Actual vs Predicted Sales"
)

plt.grid(True)

plt.show()

#  ADVERTISING IMPACT ANALYSIS

advertising_impact = data[
    [
        "TV",
        "Radio",
        "Newspaper",
        "Sales"
    ]
].corr()

print(
    "\nAdvertising Correlation Analysis"
)

print(advertising_impact)


#  FEATURE IMPORTANCE

coefficients = pipeline.named_steps[
    "model"
].coef_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Impact": coefficients
})

importance_df = importance_df.sort_values(
    by="Impact",
    ascending=False
)

print("\nMarketing Channel Impact")
print(importance_df)

plt.figure(figsize=(8,5))

plt.bar(
    importance_df["Feature"],
    importance_df["Impact"]
)

plt.title(
    "Advertising Channel Impact on Sales"
)

plt.ylabel(
    "Coefficient Value"
)

plt.show()

#  FUTURE SALES FORECAST

future_campaign = pd.DataFrame({
    "TV": [250],
    "Radio": [35],
    "Newspaper": [50]
})

future_sales = pipeline.predict(
    future_campaign
)

print(
    "\nPredicted Sales For Future Campaign:",
    round(
        future_sales[0],
        2
    )
)

# SAVE MODEL

joblib.dump(
    pipeline,
    "sales_prediction_model.pkl"
)

print(
    "\nModel Saved Successfully"
)

# BUSINESS INSIGHTS


print("""
BUSINESS INSIGHTS

1. Advertising spending directly impacts sales.

2. TV and Radio campaigns usually generate
   stronger sales influence.

3. Companies can optimize marketing budget
   using predictive analytics.

4. Sales forecasting helps improve inventory
   and business planning.

5. Data-driven marketing reduces unnecessary
   advertising expenses.
""")

