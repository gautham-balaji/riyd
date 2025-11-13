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
