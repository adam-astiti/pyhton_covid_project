# Japan COVID-19 Pandemic: An End-to-End Analysis and Forecasting Project
This repository contains a comprehensive, end-to-end data analysis project that explores the dynamics of the COVID-19 pandemic in Japan. The project showcases a full data science workflow, from initial data cleaning and exploratory analysis to building and deploying a predictive machine learning model.

 ![covid](https://github.com/user-attachments/assets/d0f5b7a5-4069-434d-b319-9854b00c87c6)
Click here for: [Live Streaming of Project](https://adampyhtoncovidproject.streamlit.app/)

# Project Introduction

The COVID-19 pandemic has been one of the most significant global health crises in modern history, generating an unprecedented amount of data. For data analysts, this provides a unique opportunity to apply analytical techniques to understand and interpret the complex dynamics of a real-world pandemic.

This project undertakes an end-to-end analysis of the COVID-19 pandemic in Japan, utilizing the comprehensive dataset provided by Our World in Data. With a background in biology, this analysis goes beyond surface-level metrics to explore the epidemiological and immunological narratives hidden within the data.

## The primary objectives of this project are:

1. To conduct a thorough exploratory data analysis (EDA) to visualize and understand the key trends, including the evolution of infection waves and the impact of the national vaccination campaign.
   
2. To develop and evaluate supervised machine learning models to forecast future cases and classify weekly risk levels, demonstrating the practical application of predictive analytics in public health.

Ultimately, this project serves as a comprehensive case study demonstrating the ability to handle complex, real-world data, derive meaningful insights, and build predictive models to answer critical questions.

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

Of course. Highlighting your use of interactive Plotly is a great idea as it showcases more advanced visualization skills.

Here is the revised text for that section of your README.

# Streamlit Application Files
These files work together to create the interactive web dashboard.

## Main_Page.py
![covid page 1](https://github.com/user-attachments/assets/0d329fa4-244a-4969-8e27-8d91df276e5e)

This is the main entry point and the "home page" of the Streamlit application.

- Purpose: To introduce the project and display the key findings from the Exploratory Data Analysis (EDA). This page uses interactive Plotly charts to create a dynamic and engaging user experience, setting the context before the user explores the predictive model.

## pages/Machine_Learning.py
![covid page 2](https://github.com/user-attachments/assets/825763dc-5163-4097-b204-6db0cbcc0f10)

This script creates the second page of the application, dedicated to the machine learning model.

- Purpose: To provide an interactive interface for the forecasting model. It displays the model's performance metrics (like the R² score) and allows users to run a 7-day forecast, complete with a visualization of the results and an uncertainty interval.

## utils.py
This is a helper script that contains reusable functions to keep the main application code clean and organized.

- Purpose: To store functions for tasks that are used in multiple places, such as loading data, loading the model, and the core logic for generating the iterative forecast. This is a best practice for writing modular and maintainable code.
