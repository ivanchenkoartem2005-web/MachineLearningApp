import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    adjusted_rand_score
)

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

digits = load_digits()

X = digits.data
y = digits.target

df = pd.DataFrame(X)
df["Digit"] = y

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())

print("\nDigit distribution:")
print(df["Digit"].value_counts().sort_index())

print("\nAverage pixel intensity:", X.mean())
print("Maximum pixel intensity:", X.max())
print("Minimum pixel intensity:", X.min())
print("Standard deviation:", X.std())

print("\nAverage pixel intensity per digit:")
print(df.groupby("Digit").mean().mean(axis=1))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Digit: {digits.target[i]}")
    ax.axis("off")

plt.suptitle("Examples of Handwritten Digits")
plt.tight_layout()
plt.show()

plt.hist(y, bins=np.arange(-0.5, 10.5, 1), edgecolor="black")

plt.title("Distribution of Handwritten Digits")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.xticks(range(10))
plt.show()

correlation = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()
plt.title("Feature Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logr = LogisticRegression(max_iter=3000)

logr.fit(X_train_scaled, y_train)

logr_prediction = logr.predict(X_test_scaled)

logr_accuracy = accuracy_score(
    y_test,
    logr_prediction
)

print("\nLogistic Regression Accuracy:", logr_accuracy)

print(classification_report(y_test, logr_prediction))

cm = confusion_matrix(y_test, logr_prediction)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

knn_prediction = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(
    y_test,
    knn_prediction
)

print("\nKNN Accuracy:", knn_accuracy)

dtree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

dtree.fit(X_train, y_train)

tree_prediction = dtree.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_prediction
)

print("\nDecision Tree Accuracy:", tree_accuracy)


plt.figure(figsize=(18, 10))

plot_tree(
    dtree,
    max_depth=3,
    filled=True
)

plt.title("Decision Tree Visualization")
plt.show()

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "K Nearest Neighbors",
        "Decision Tree"
    ],
    "Accuracy": [
        logr_accuracy,
        knn_accuracy,
        tree_accuracy
    ]
})

print("\nModel Comparison:")
print(results)

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.show()

random_index = 10

sample = X_test[random_index].reshape(1, -1)

prediction = logr.predict(
    scaler.transform(sample)
)

plt.imshow(
    sample.reshape(8, 8),
    cmap="gray"
)

plt.title(f"Predicted digit: {prediction[0]}")
plt.axis("off")
plt.show()

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

cluster_labels = kmeans.labels_

print("\nKMeans Inertia:", kmeans.inertia_)

print(
    "Adjusted Rand Score:",
    adjusted_rand_score(y_train, cluster_labels)
)

inertias = []

for i in range(1, 16):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_train_scaled)

    inertias.append(model.inertia_)


plt.plot(
    range(1, 16),
    inertias,
    marker="o"
)

plt.title("KMeans Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="tab10",
    alpha=0.7
)

plt.colorbar(scatter)
plt.title("Digit Distribution Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

sample_data = X_train_scaled[:100]

linked = linkage(
    sample_data,
    method="ward"
)

plt.figure(figsize=(14, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

agg = AgglomerativeClustering(
    n_clusters=10
)

agg_labels = agg.fit_predict(sample_data)

print(
    "\nHierarchical Clustering ARI:",
    adjusted_rand_score(y_train[:100], agg_labels)
)


print("\nProject completed!")
