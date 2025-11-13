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
