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
