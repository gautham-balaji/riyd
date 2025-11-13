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
