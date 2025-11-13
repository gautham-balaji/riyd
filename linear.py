import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

# DATA: X_train, X_test, y_train, y_test assumed available

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"\n=== {name} ===")
    print(f"R2: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}")
    plt.figure()
    plt.scatter(y_test, pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Predicted vs Actual")
    plt.show()

    # metrics df for this model (created AFTER the model)
    metrics_df = pd.DataFrame(columns=["model", "r2", "mse", "mae", "notes"])
    metrics_df.loc[len(metrics_df)] = [name, r2, mse, mae, ""]
    print("\nMetrics DataFrame:")
    print(metrics_df)

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
pred = poly_model.predict(X_test_poly)
r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print("\n=== Polynomial (degree=2) LinearRegression ===")
print(f"R2: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}")
plt.figure()
plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Poly (deg=2) Predicted vs Actual")
plt.show()

metrics_df = pd.DataFrame(columns=["model", "r2", "mse", "mae", "notes"])
metrics_df.loc[len(metrics_df)] = ["Polynomial(deg2)_LinearRegression", r2, mse, mae, ""]
print("\nMetrics DataFrame:")
print(metrics_df)
