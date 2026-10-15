
import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Load dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(iris.target).map(
    dict(enumerate(iris.target_names))
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Dashboard heading
st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the flower measurements below "
    "to predict its species using the KNN algorithm."
)

st.divider()

# Display project information
col1, col2, col3 = st.columns(3)

col1.metric("Total Flowers", "150")
col2.metric("Algorithm", "KNN")
col3.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.divider()

# Input section
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# Prediction
if st.button("Predict Flower", type="primary"):

    new_flower = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=X.columns
    )

    prediction = model.predict(new_flower)[0]

    st.success(f"Predicted Species: {prediction.title()}")

    st.write("Entered Measurements:")
    st.dataframe(new_flower, use_container_width=True)

st.divider()

st.caption(
    "Iris Flower Classification Project | "
    "Machine Learning using K-Nearest Neighbors"
)
