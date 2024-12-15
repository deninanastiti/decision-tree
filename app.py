import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from io import StringIO

# Set the page title and layout
st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

# Title with some description
st.title("📊 Decision Tree Classifier & Visualizer")
st.markdown("""
    This app allows you to upload a dataset, train a Decision Tree model, 
    and visualize the decision tree structure. You can also download the 
    dataset as a CSV after processing it.
""")

# Add some custom CSS for a better appearance
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# File uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# When file is uploaded
if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)

    # Display basic dataset info
    st.sidebar.write("Dataset Preview")
    st.sidebar.dataframe(data.head())

    # Select target and feature columns
    st.sidebar.header("Select Features and Target")
    target_column = st.sidebar.selectbox("Target Column", data.columns)
    feature_columns = st.sidebar.multiselect("Feature Columns", data.columns)

    if target_column and len(feature_columns) > 0:
        # Preparing the data
        X = data[feature_columns]
        y = data[target_column]

        # Display a progress bar while training
        st.markdown("### Training Model... ⏳")
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)

        # Split the data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Show accuracy
        accuracy = clf.score(X_test, y_test)
        st.write(f"**Model Accuracy**: {accuracy * 100:.2f}%")

        # Visualization Section
        st.subheader("Decision Tree Visualization")

        # Graphviz visualization
        dot_data = tree.export_graphviz(clf, out_file=None, 
                                       feature_names=feature_columns,  
                                       class_names=np.unique(y).astype(str),  
                                       filled=True, rounded=True,  
                                       special_characters=True)  
        graph = graphviz.Source(dot_data)  
        st.graphviz_chart(dot_data)

        # Plot Tree using Matplotlib
        st.subheader("Decision Tree Plot")
        fig, ax = plt.subplots(figsize=(12, 12))
        tree.plot_tree(clf, feature_names=feature_columns, class_names=np.unique(y).astype(str),
                       filled=True, fontsize=10, ax=ax)
        st.pyplot(fig)

