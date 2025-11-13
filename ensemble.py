import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
import pandas as pd

is_classification = len(np.unique(y_train)) <= 20 and y_train.dtype.kind in 'iu'

if is_classification:
    # AdaBoost
    params = {"n_estimators":[50,100], "learning_rate":[0.5,1.0]}
    grid = GridSearchCV(AdaBoostClassifier(random_state=42), params, cv=3, scoring="accuracy")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    pred = best.predict(X_test)
    print("\n=== AdaBoost (best) ===")
    print("Best params:", grid.best_params_)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="macro", zero_division=0)
    rec = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    roc_auc = None
    try:
        probs = best.predict_proba(X_test)
        if probs is not None and probs.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, probs[:,1])
    except Exception:
        roc_auc = None

    metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
    metrics_df.loc[len(metrics_df)] = ["AdaBoost(GridSearch)", acc, prec, rec, f1, roc_auc, str(grid.best_params_)]
    print("\nMetrics DataFrame:")
    print(metrics_df)

    # GradientBoosting
    params = {"n_estimators":[50,100], "learning_rate":[0.1,0.5], "max_depth":[3,5]}
    grid2 = GridSearchCV(GradientBoostingClassifier(random_state=42), params, cv=3, scoring="accuracy")
    grid2.fit(X_train, y_train)
    best2 = grid2.best_estimator_
    pred2 = best2.predict(X_test)
    print("\n=== GradientBoosting (best) ===")
    print("Best params:", grid2.best_params_)
    print(confusion_matrix(y_test, pred2))
    print(classification_report(y_test, pred2))

    acc = accuracy_score(y_test, pred2)
    prec = precision_score(y_test, pred2, average="macro", zero_division=0)
    rec = recall_score(y_test, pred2, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred2, average="macro", zero_division=0)
    roc_auc = None
    try:
        probs2 = best2.predict_proba(X_test)
        if probs2 is not None and probs2.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, probs2[:,1])
    except Exception:
        roc_auc = None

    metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
    metrics_df.loc[len(metrics_df)] = ["GradientBoosting(GridSearch)", acc, prec, rec, f1, roc_auc, str(grid2.best_params_)]
    print("\nMetrics DataFrame:")
    print(metrics_df)

else:
    params = {"n_estimators":[50,100], "learning_rate":[0.1,0.5]}
    grid = GridSearchCV(AdaBoostRegressor(random_state=42), params, cv=3, scoring="r2")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    pred = best.predict(X_test)
    print("\n=== AdaBoostRegressor (best) ===")
    print("Best params:", grid.best_params_)
    print(f"R2: {r2_score(y_test, pred):.4f}  MSE: {mean_squared_error(y_test, pred):.4f}")

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = None
    try:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, pred)
    except Exception:
        mae = None

    metrics_df = pd.DataFrame(columns=["model", "r2", "mse", "mae", "notes"])
    metrics_df.loc[len(metrics_df)] = ["AdaBoostRegressor(GridSearch)", r2, mse, mae, str(grid.best_params_)]
    print("\nMetrics DataFrame:")
    print(metrics_df)

    params2 = {"n_estimators":[50,100], "learning_rate":[0.1,0.5], "max_depth":[3,5]}
    grid2 = GridSearchCV(GradientBoostingRegressor(random_state=42), params2, cv=3, scoring="r2")
    grid2.fit(X_train, y_train)
    best2 = grid2.best_estimator_
    pred2 = best2.predict(X_test)
    print("\n=== GradientBoostingRegressor (best) ===")
    print("Best params:", grid2.best_params_)
    print(f"R2: {r2_score(y_test, pred2):.4f}  MSE: {mean_squared_error(y_test, pred2):.4f}")

    r2 = r2_score(y_test, pred2)
    mse = mean_squared_error(y_test, pred2)
    mae = None
    try:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, pred2)
    except Exception:
        mae = None

    metrics_df = pd.DataFrame(columns=["model", "r2", "mse", "mae", "notes"])
    metrics_df.loc[len(metrics_df)] = ["GradientBoostingRegressor(GridSearch)", r2, mse, mae, str(grid2.best_params_)]
    print("\nMetrics DataFrame:")
    print(metrics_df)
