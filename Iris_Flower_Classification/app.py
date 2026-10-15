
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert the dataset into a DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add flower species names
df["Species"] = [
    iris.target_names[i] for i in iris.target
]

# Display the first five rows
print(df.head())
# Display the number of rows and columns
print("Dataset Shape:", df.shape)

# Display column names
print("Column Names:", df.columns)

# Display the number of missing values
print("Missing Values:")
print(df.isnull().sum())

# Display the number of flowers in each species
print("Species Count:")
print(df["Species"].value_counts())
import matplotlib.pyplot as plt

# Scatter plot for petal measurements
plt.figure(figsize=(8, 5))

for species in df["Species"].unique():
    data = df[df["Species"] == species]

    plt.scatter(
        data["petal length (cm)"],
        data["petal width (cm)"],
        label=species
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")
plt.legend()
plt.grid(True)

plt.show()

# Scatter plot for sepal measurements
plt.figure(figsize=(8, 5))

for species in df["Species"].unique():
    data = df[df["Species"] == species]

    plt.scatter(
        data["sepal length (cm)"],
        data["sepal width (cm)"],
        label=species
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")
plt.legend()
plt.grid(True)

plt.show()

from sklearn.model_selection import train_test_split

# Input features: flower measurements
X = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]

# Target: flower species
y = df["Species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the results
print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Create the model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

# Train the model
model.fit(X_train, y_train)

# Predict flower species
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Display performance report
print(classification_report(y_test, y_pred))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("KNN Confusion Matrix")
plt.show()

# Predict a new flower

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("New Flower Measurements:", new_flower)
print("Predicted Flower Species:", prediction[0])
