# --------------------------------------------------------------------------
# MASTER MODEL SNIPPETS FILE
# --------------------------------------------------------------------------
# NOTE:
# - This file assumes you have already run your preprocessing script and that
#   these variables are available in the environment:
#       X_train, X_test, y_train, y_test, final_features, df_scaled
# - Each snippet has its own imports and its own metrics_df (created after the model).
# - Plots are shown with plt.show()
# - You mentioned you will run snippets selectively; run only the parts you want.
# --------------------------------------------------------------------------

# -------------------------
# 1) Regression models: Linear, Ridge, Lasso, Polynomial
# -------------------------
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


# -------------------------
# 2) Naive Bayes variants: CategoricalNB, GaussianNB, MultinomialNB, BernoulliNB
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd

nb_models = {
    "CategoricalNB": CategoricalNB(),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB()
}

classes = np.unique(y_train)
y_test_bin = None
if len(classes) > 2:
    y_test_bin = label_binarize(y_test, classes=classes)

for name, model in nb_models.items():
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"\n{name} failed to fit: {e}")
        continue
    preds = model.predict(X_test)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))

    # compute metrics for DataFrame
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    roc_auc = None
    if probs is not None:
        if len(classes) == 2:
            try:
                roc_auc = roc_auc_score(y_test, probs[:, 1])
                fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                print(f"ROC-AUC: {roc_auc:.4f}")
                plt.figure(); plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title(f"{name} ROC Curve"); plt.show()
            except Exception as e:
                print("ROC error:", e)
        else:
            try:
                roc_auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")
                print(f"Macro ROC-AUC: {roc_auc:.4f}")
                for i, cls in enumerate(classes[:3]):  # show up to 3 class ROC plots
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                    plt.plot(fpr, tpr, label=f"class {cls}")
                plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title(f"{name} ROC Curve (sample classes)"); plt.show()
            except Exception as e:
                print("ROC multi error:", e)

    metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
    metrics_df.loc[len(metrics_df)] = [name, acc, prec, rec, f1, roc_auc, ""]
    print("\nMetrics DataFrame:")
    print(metrics_df)


# -------------------------
# 3) Logistic Regression — binary and multiclass
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

print("\n=== LogisticRegression ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Classification Report:")
print(classification_report(y_test, preds))

classes = np.unique(y_train)
roc_auc = None
if probs is not None:
    if len(classes) == 2:
        try:
            roc_auc = roc_auc_score(y_test, probs[:, 1])
            fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
            print(f"ROC-AUC: {roc_auc:.4f}")
            plt.figure(); plt.plot(fpr, tpr, label=f"Logistic (AUC={roc_auc:.3f})"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("Logistic ROC"); plt.show()
        except Exception as e:
            print("ROC error:", e)
    else:
        try:
            y_test_bin = label_binarize(y_test, classes=classes)
            roc_auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")
            print(f"Macro ROC-AUC: {roc_auc:.4f}")
            for i, cls in enumerate(classes[:3]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                plt.plot(fpr, tpr, label=f"class {cls}")
            plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("Logistic ROC (sample classes)"); plt.show()
        except Exception as e:
            print("ROC multi error:", e)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="macro", zero_division=0)
rec = recall_score(y_test, preds, average="macro", zero_division=0)
f1 = f1_score(y_test, preds, average="macro", zero_division=0)

metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
metrics_df.loc[len(metrics_df)] = ["LogisticRegression", acc, prec, rec, f1, roc_auc, ""]
print("\nMetrics DataFrame:")
print(metrics_df)


# -------------------------
# 4) Perceptron (single-layer) & MLP
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd

# Perceptron
p = Perceptron(max_iter=1000)
p.fit(X_train, y_train)
pred_p = p.predict(X_test)
print("\n=== Perceptron ===")
print(confusion_matrix(y_test, pred_p))
print(classification_report(y_test, pred_p))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_test, pred_p)
prec = precision_score(y_test, pred_p, average="macro", zero_division=0)
rec = recall_score(y_test, pred_p, average="macro", zero_division=0)
f1 = f1_score(y_test, pred_p, average="macro", zero_division=0)
metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "notes"])
metrics_df.loc[len(metrics_df)] = ["Perceptron", acc, prec, rec, f1, ""]
print("\nMetrics DataFrame:")
print(metrics_df)

# MLP (multi-layer perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)
probs_mlp = mlp.predict_proba(X_test) if hasattr(mlp, "predict_proba") else None
print("\n=== MLPClassifier ===")
print(confusion_matrix(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp))

roc_auc = None
classes = np.unique(y_train)
if probs_mlp is not None:
    if len(classes) == 2:
        try:
            roc_auc = roc_auc_score(y_test, probs_mlp[:, 1])
            fpr, tpr, _ = roc_curve(y_test, probs_mlp[:, 1])
            print(f"MLP ROC-AUC: {roc_auc:.4f}")
            plt.figure(); plt.plot(fpr, tpr, label=f"MLP (AUC={roc_auc:.3f})"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("MLP ROC"); plt.show()
        except Exception as e:
            print("ROC error:", e)
    else:
        try:
            y_test_bin = label_binarize(y_test, classes=classes)
            roc_auc = roc_auc_score(y_test_bin, probs_mlp, average="macro", multi_class="ovr")
            print(f"MLP Macro ROC-AUC: {roc_auc:.4f}")
            for i, cls in enumerate(classes[:3]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_mlp[:, i])
                plt.plot(fpr, tpr, label=f"class {cls}")
            plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("MLP ROC (sample classes)"); plt.show()
        except Exception as e:
            print("ROC multi error:", e)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_test, pred_mlp)
prec = precision_score(y_test, pred_mlp, average="macro", zero_division=0)
rec = recall_score(y_test, pred_mlp, average="macro", zero_division=0)
f1 = f1_score(y_test, pred_mlp, average="macro", zero_division=0)
metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
metrics_df.loc[len(metrics_df)] = ["MLPClassifier", acc, prec, rec, f1, roc_auc, ""]
print("\nMetrics DataFrame:")
print(metrics_df)


# -------------------------
# 5) SVM with multiple kernels: rbf, sigmoid, poly, linear
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd

kernels = ["rbf", "sigmoid", "poly", "linear"]
classes = np.unique(y_train)
y_test_bin = None
if len(classes) > 2:
    y_test_bin = label_binarize(y_test, classes=classes)

for kernel in kernels:
    clf = SVC(kernel=kernel, probability=True)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    print(f"\n=== SVM kernel={kernel} ===")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    roc_auc = None
    if len(classes) == 2:
        try:
            roc_auc = roc_auc_score(y_test, probs[:, 1])
            fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
            print(f"ROC-AUC: {roc_auc:.4f}")
            plt.figure(); plt.plot(fpr, tpr, label=f"SVM-{kernel} (AUC={roc_auc:.3f})"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title(f"SVM ({kernel}) ROC"); plt.show()
        except Exception as e:
            print("ROC error:", e)
    else:
        try:
            roc_auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")
            print(f"Macro ROC-AUC: {roc_auc:.4f}")
            for i, cls in enumerate(classes[:3]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                plt.plot(fpr, tpr, label=f"class {cls}")
            plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title(f"SVM ({kernel}) ROC (sample)"); plt.show()
        except Exception as e:
            print("ROC multi error:", e)

    metrics_df = pd.DataFrame(columns=["model", "kernel", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "notes"])
    metrics_df.loc[len(metrics_df)] = ["SVM", kernel, acc, prec, rec, f1, roc_auc, ""]
    print("\nMetrics DataFrame:")
    print(metrics_df)


# -------------------------
# 6) Decision Tree — entropy and gini (CART)
# -------------------------
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


# -------------------------
# 7) Clustering — KMeans, DBSCAN, KModes, Agglomerative
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

# Use X_all = stacked train+test
X_all = np.vstack([X_train, X_test])

# KMeans (default init='k-means++')
for k in [2, 3, 5]:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = km.fit_predict(X_all)
    sil = silhouette_score(X_all, labels) if len(np.unique(labels))>1 else -1
    print(f"KMeans k={k} silhouette={sil:.4f}")

    metrics_df = pd.DataFrame(columns=["model", "k", "silhouette", "notes"])
    metrics_df.loc[len(metrics_df)] = ["KMeans", k, sil, ""]
    print("\nMetrics DataFrame:")
    print(metrics_df)

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_all)
unique = np.unique(labels)
print(f"DBSCAN found clusters: {unique}")
sil = None
try:
    sil = silhouette_score(X_all, labels) if len(np.unique(labels))>1 else -1
except Exception:
    sil = None

metrics_df = pd.DataFrame(columns=["model", "silhouette", "notes"])
metrics_df.loc[len(metrics_df)] = ["DBSCAN", sil, ""]
print("\nMetrics DataFrame:")
print(metrics_df)

# KModes (categorical clustering, only if available and your features are categorical encoded)
try:
    from kmodes.kmodes import KModes
    kmodes_available = True
except Exception:
    kmodes_available = False

if kmodes_available:
    try:
        kmode = KModes(n_clusters=3, init='Huang', random_state=42)
        lab = kmode.fit_predict(X_all.astype('int'))
        print("KModes cluster sizes:", np.bincount(lab))
        metrics_df = pd.DataFrame(columns=["model", "notes"])
        metrics_df.loc[len(metrics_df)] = ["KModes", "ran successfully"]
        print("\nMetrics DataFrame:")
        print(metrics_df)
    except Exception as e:
        print("KModes failed:", e)
        metrics_df = pd.DataFrame(columns=["model", "notes"])
        metrics_df.loc[len(metrics_df)] = ["KModes", f"failed: {e}"]
        print("\nMetrics DataFrame:")
        print(metrics_df)
else:
    print("KModes not available. pip install kmodes to enable it.")

# Agglomerative
for k in [2,3,5]:
    agg = AgglomerativeClustering(n_clusters=k)
    lab = agg.fit_predict(X_all)
    print(f"Agglomerative n_clusters={k} unique={np.unique(lab)}")
    sil = None
    try:
        sil = silhouette_score(X_all, lab) if len(np.unique(lab))>1 else -1
    except Exception:
        sil = None
    metrics_df = pd.DataFrame(columns=["model", "k", "silhouette", "notes"])
    metrics_df.loc[len(metrics_df)] = ["Agglomerative", k, sil, ""]
    print("\nMetrics DataFrame:")
    print(metrics_df)


# -------------------------
# 8) Ensemble — AdaBoost and GradientBoost (GridSearchCV)
# -------------------------
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


# -------------------------
# 9) Dimensionality Reduction — PCA and t-SNE
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


# Use X_all (stacked train+test)
X_all = np.vstack([X_train, X_test])
labels = None
try:
    labels = np.hstack([y_train, y_test])
except Exception:
    labels = None


pca = PCA(n_components=0.95)
pc_full = pca.fit_transform(X_all)

# Metrics DF for PCA (full dimensionality)
metrics_df = pd.DataFrame(columns=["model", "variance_covered", "n_components", "notes"])
metrics_df.loc[len(metrics_df)] = ["PCA_95", 0.95, pca.n_components_, ""]
print("\nMetrics DataFrame (PCA 95%):")
print(metrics_df)

X_pca = pc_full   # use PCA features
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# PCA 2D for visualization ONLY
pca2 = PCA(n_components=2)
pc_2d = pca2.fit_transform(X_all)

plt.figure()
if labels is not None:
    plt.scatter(pc_2d[:,0], pc_2d[:,1], c=labels, cmap='viridis', alpha=0.7)
else:
    plt.scatter(pc_2d[:,0], pc_2d[:,1], alpha=0.7)
plt.title("PCA (2D Visualization)")
plt.show()

# t-SNE (can be slow)
tsne = TSNE(n_components=2, perplexity=30, n_iter=800, random_state=42)
tx = tsne.fit_transform(X_all)
plt.figure()
if labels is not None:
    plt.scatter(tx[:,0], tx[:,1], c=labels, cmap='viridis', alpha=0.7)
else:
    plt.scatter(tx[:,0], tx[:,1], alpha=0.7)
plt.title("t-SNE (2D)")
plt.show()

metrics_df = pd.DataFrame(columns=["model", "notes"])
metrics_df.loc[len(metrics_df)] = ["t-SNE_2D", "visualized", ""]
print("\nMetrics DataFrame:")
print(metrics_df)


# -------------------------
# 10) Record metrics snippet example (kept as separate snippet)
# -------------------------
import pandas as pd

# This is just an example snippet showing how to create/print a small metrics df.
metrics_df = pd.DataFrame(columns=["model","type","metric1","metric2","metric3","notes"])
print("Current example metrics_df (empty):")
print(metrics_df)

# Example append usage (not executed model):
# metrics_df.loc[len(metrics_df)] = ["ExampleModel", "classification", "accuracy:0.95", "f1:0.94", "", "sample note"]
# print(metrics_df)

# --------------------------------------------------------------------------
# END OF FILE
# --------------------------------------------------------------------------
