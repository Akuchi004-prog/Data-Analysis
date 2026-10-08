import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Load Dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["Species"] = iris.target_names[iris.target]

print("DATASET OVERVIEW")
print(f"Total samples: {df.shape[0]}, Features: {df.shape[1]-1}")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Summary Statistics
print("\nSUMMARY STATISTICS")
print(df.describe())
print("\nStatistics grouped by Species:")
print(df.groupby("Species").mean())

# 3. Visualizations
# A) Pair Plot
sns.pairplot(df, hue="Species", diag_kind="kde")
plt.suptitle("Pairwise Relationships by Iris Species", y=1.03)
plt.show()

# B) Boxplots for each feature
features = df.columns[:-1]
plt.figure(figsize=(14, 8))
for idx, feat in enumerate(features, 1):
    plt.subplot(2, 2, idx)
    sns.boxplot(data=df, x="Species", y=feat)
    plt.title(f"{feat} by Species")
plt.tight_layout()
plt.show()

# C) Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nWEEK 11 EDA COMPLETE — all plots displayed")



# WEEK 12: Decision Tree Classifier

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Prepare features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print("\nDECISION TREE MODEL — WEEK 12")

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")

# Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("\nPERFORMANCE METRICS")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix")
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(14, 9))
plot_tree(model, feature_names=X.columns, class_names=model.classes_,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree — Iris Classification", fontsize=14)
plt.show()

print("\nALL DONE — Week 11 & 12 Complete!")