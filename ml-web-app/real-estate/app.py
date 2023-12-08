import streamlit as st
from regpredict_page import show_regpredict_page
from regexplore_page import show_regexplore_page

page = st.sidebar.selectbox("Model Type", ("Classification", "Regression"))

if page == "Classification":
    # Code for classification model
    subPage = st.sidebar.selectbox("Classification Model", ("Predict", "Explore"))
    if subPage == "Predict":
        # Dynamically load and run the script for classification prediction
        exec(open(r"C:\Users\ACER\ITD105\ml-web-app\email-spam\App\classpredict_page.py").read())
        pass
    else:
        # Dynamically load and run the script for classification prediction
        exec(open(r"C:\Users\ACER\ITD105\ml-web-app\email-spam\App\classexplore_page.py").read())
        pass
else:
    # Code for regression model
    subPage = st.sidebar.selectbox("Regression Model", ("Predict", "Explore"))
    if subPage == "Predict":
        show_regpredict_page()
        pass
    else:
        show_regexplore_page()
        pass
