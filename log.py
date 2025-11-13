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
