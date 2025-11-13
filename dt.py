import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error
import pandas as pd

# Heuristic to choose classifier vs regressor
if len(np.unique(y_train)) <= 20 and y_train.dtype.kind in 'iu':
    for crit in ["gini", "entropy"]:
        dt = DecisionTreeClassifier(criterion=crit)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        print(f"\n=== DecisionTree (criterion={crit}) ===")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        metrics_df = pd.DataFrame(columns=["model", "criterion", "accuracy", "precision_macro", "recall_macro", "f1_macro", "notes"])
        metrics_df.loc[len(metrics_df)] = ["DecisionTreeClassifier", crit, acc, prec, rec, f1, ""]
        print("\nMetrics DataFrame:")
        print(metrics_df)
else:
    for crit in ["squared_error", "friedman_mse"]:
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        pred = dt.predict(X_test)
        print(f"\n=== DecisionTreeRegressor (criterion={crit}) ===")
        print(f"R2: {r2_score(y_test, pred):.4f}  MSE: {mean_squared_error(y_test, pred):.4f}")
        plt.figure(); plt.scatter(y_test, pred, alpha=0.6); plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--'); plt.title(f"DT Regressor ({crit}) Pred vs Actual"); plt.show()

        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        mae = None
        try:
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_test, pred)
        except Exception:
            mae = None

        metrics_df = pd.DataFrame(columns=["model", "criterion", "r2", "mse", "mae", "notes"])
        metrics_df.loc[len(metrics_df)] = ["DecisionTreeRegressor", crit, r2, mse, mae, ""]
        print("\nMetrics DataFrame:")
        print(metrics_df)
