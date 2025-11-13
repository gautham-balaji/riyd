import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================
# HARD-CODE YOUR DATASET HERE
# ============================
dataset_path = "your_dataset.csv"
target = "target"

df = pd.read_csv(dataset_path)
# df_test = pd.read_csv(test)

# ============================
# BASIC EDA
# ============================
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe(include="all"))

# ============================
# DROP MISSING VALUES
# ============================
df = df.dropna()

# ============================
# UNIVARIATE ANALYSIS
# ============================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target]

categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n===== HISTOGRAMS FOR NUMERIC COLUMNS =====")
for col in numeric_cols:
    plt.figure()
    plt.title(f"Histogram of {col}")
    plt.hist(df[col], bins=30)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

print("\n===== VALUE COUNTS FOR CATEGORICAL COLUMNS =====")
for col in categorical_cols:
    plt.figure()
    plt.title(f"Value Counts of {col}")
    df[col].value_counts().plot(kind='bar')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# ============================
# LABEL ENCODE CATEGORICALS
# ============================
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Update numeric list after encoding
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target]

# ============================
# CORRELATION HEATMAP
# ============================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================
# SELECT TOP 5 NUMERIC FEATURES BY CORRELATION WITH TARGET
# ============================
correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
top5 = correlations.head(5).index.tolist()

print("\n===== TOP 5 NUMERIC FEATURES (BY CORR WITH TARGET) =====")
print(correlations.head(5))

# ============================
# SCATTER PLOTS FOR TOP 5
# ============================
for col in top5:
    plt.figure()
    plt.title(f"{col} vs {target}")
    plt.scatter(df[col], df[target])
    plt.xlabel(col)
    plt.ylabel(target)
    plt.show()

# ============================
# CORRELATION-BASED FEATURE SELECTION
# Step 1: keep features with |corr| ≥ 0.1
# ============================
strong_features = correlations[correlations >= 0.1].index.tolist()

print("\n===== FEATURES WITH CORR ≥ 0.1 =====")
print(strong_features)

# ============================
# MULTICOLLINEARITY REMOVAL
# Remove features correlated ≥ 0.85 with each other
# Keep the one with stronger corr with target
# ============================
corr_matrix = df[strong_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

print("\n===== MULTICOLLINEAR FEATURE PAIRS (corr ≥ 0.85) =====")
for col in upper.columns:
    for row in upper.index:
        if upper.loc[row, col] >= 0.85:
            print(f"{row}  <-->  {col}  (corr = {upper.loc[row, col]:.3f})")

final_features = strong_features  # keep everything


print("\n===== FINAL SELECTED FEATURES =====")
print(final_features)

# ============================
# STANDARD SCALING
# ============================
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[final_features] = scaler.fit_transform(df_scaled[final_features])

# ============================
# TRAIN TEST SPLIT
# ============================
X = df_scaled[final_features]
y = df_scaled[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = df_scaled[final_features]
# y_train = df_scaled[target]
# X_test = df_test[final_features]
# y_test = df_test[target]
print("\n===== TRAIN/TEST SHAPES =====")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n===== DONE =====")
