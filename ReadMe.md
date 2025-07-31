# Japan COVID-19 Pandemic: An End-to-End Analysis and Forecasting Project
This repository contains a comprehensive, end-to-end data analysis project that explores the dynamics of the COVID-19 pandemic in Japan. The project showcases a full data science workflow, from initial data cleaning and exploratory analysis to building and deploying a predictive machine learning model.

➡️ View the Live Streamlit Dashboard Here 👈 (Don't forget to add your Streamlit app link here!)

# 📂 Project Structure & File Explanations
This project is organized into distinct files and folders, each with a specific purpose.

## EDA.ipynb - Exploratory Data Analysis
This Jupyter Notebook is where the initial exploration and storytelling take place. The primary goal of this notebook is to "get to know" the data and uncover the core narratives of the pandemic.

### Key activities in this file:

- Data Cleaning: Initial loading of the raw dataset and handling of inconsistencies like missing values and incorrect data types.

- Visual Analysis: Creation of key visualizations to understand:

- The evolution of pandemic waves over time.

- The impact of the national vaccination campaign on fatality rates.

- The relationship between government policies (stringency_index) and the virus's transmission rate.

## ML Model.ipynb - Machine Learning Model Creation
This notebook documents the process of building, training, and validating the predictive model. It serves as the "research and development" phase for the forecasting engine.

### Key activities in this file:

- Feature Engineering: Creation of historical lag and rolling window features, which are essential for time-series forecasting.

- Model Selection & Validation: A Ridge regression model was chosen for its robustness. It was rigorously validated using Time-Series Cross-Validation to ensure its performance is stable and reliable across different periods of the pandemic.

- Model Saving: The final, trained model is saved to a .pkl file so it can be easily loaded into the Streamlit application without needing to be retrained.

## Streamlit Application Files
These files work together to create the interactive web dashboard.

### Main_Page.py
This is the main entry point and the "home page" of the Streamlit application.

Purpose: To introduce the project to the user and to display the key findings from the Exploratory Data Analysis (EDA). It sets the context before the user explores the predictive model.

### pages/Machine_Learning.py
This script creates the second page of the application, dedicated to the machine learning model.

Purpose: To provide an interactive interface for the forecasting model. It displays the model's performance metrics (like the R² score) and allows users to run a 7-day forecast, complete with a visualization of the results and an uncertainty interval.

### utils.py
This is a helper script that contains reusable functions to keep the main application code clean and organized.

Purpose: To store functions for tasks that are used in multiple places, such as loading data, loading the model, and the core logic for generating the iterative forecast. This is a best practice for writing modular and maintainable code.
