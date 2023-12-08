import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def train_classification_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def show_classexplore_page():
    st.title("Explore Classification Model")

    df = load_data()
    clf, X_test, y_test = train_classification_model(df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, clf.predict(X_test))
    st.write(cm)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, clf.predict(X_test), target_names=load_iris().target_names)
    st.text(report)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.Series(clf.feature_importances_, index=df.columns[:-1])
    st.bar_chart(feature_importance)

    # Pairplot
    st.subheader("Pairplot")
    sns.pairplot(df, hue="target")
    st.pyplot()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()

# Uncomment the line below when testing this script individually
# show_classexplore_page()
