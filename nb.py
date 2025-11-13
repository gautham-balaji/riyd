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

