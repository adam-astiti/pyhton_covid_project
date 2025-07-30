import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
# Sets the title of the browser tab, adds an icon, and uses a wide layout for more space.
st.set_page_config(
    page_title="Predictive Analysis (Machine Learning)",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.image(r"copid.jpg", use_container_width=True)
import utils
df = pd.read_csv("Japan Covid Data/cleaned_data_japan-owid-covid-data.csv")

# Load css to adjust theme
utils.css_load()

# Sidebar
with st.sidebar:
    st.image('streamlit-logo-primary-colormark-darktext.png', use_container_width=False)
    st.write("""### Project by: **Adam Astiti**""")
    st.link_button('Linkedin', 'https://www.linkedin.com/in/adam-astiti-a3787312a/', icon='💼')
    st.link_button('Github', 'https://github.com/adam-astiti', icon='👨‍💻')


#Create a Function to Inspect the DataFrame
#This function will return a DataFrame with the count of non-null values, null values, amd data types
# @st.cache_data tells Streamlit to run this function only once.
@st.cache_data
def inspect_dataframe(data_frame):
    non_null_count = data_frame.count().reset_index()
    non_null_count = non_null_count.rename(columns={'index' : 'column_name', 0: 'non_null_count'})
    non_null_count['null_count'] = data_frame.shape[0] - non_null_count['non_null_count']
    column_type = data_frame.dtypes.reset_index()
    column_type = column_type.rename(columns={'index' : 'column_name', 0:'data_type'})
    dat_inspection = pd.merge(non_null_count, column_type, 'left', left_on= 'column_name', right_on='column_name')
    return dat_inspection.sort_values(by='non_null_count', ascending=False)

# Function to load model with pkl format
# @st.cache_resource is for loading resources that shouldn't be serialized, like ML models.
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found.")
        st.stop()

new_cases_prediction_model =  load_model(r'new_cases_prediction_model.pkl')

column_to_use = inspect_dataframe(df)
column_to_use	= column_to_use[column_to_use['non_null_count'] > 1000]
column_to_use = column_to_use.column_name
df = df[column_to_use]
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear

# create lagged features (7, 14, 21 days)
target_col = 'reproduction_rate'
features_to_lag = [
    'new_cases_smoothed'
]

lag_periods = [7, 14, 21]

for feature in features_to_lag:
    for lag in lag_periods:
        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

# create rolling averages
rolling_window_size = 14
feature_to_roll = 'new_cases_smoothed'

df[f'{feature_to_roll}_roll_mean_{rolling_window_size}'] = df[feature_to_roll].rolling(window=rolling_window_size).mean()
df[f'{feature_to_roll}_roll_std_{rolling_window_size}'] = df[feature_to_roll].rolling(window=rolling_window_size).std()

# Drop all null values because models cannot handle nulls
df = df.dropna()


st.markdown("""
    # Machine Learning for Predictive Forecasting
After completing the Exploratory Data Analysis (EDA), we have gained valuable insights into the historical dynamics of the COVID-19 pandemic in Japan. The next phase of this project transitions from description to prediction, leveraging supervised machine learning to create a practical forecasting tool.

**The primary objective** of this modeling phase is to forecast the number of new daily cases **(new_cases_smoothed)**.

### Why focus on New Case Prediction?
Predicting the number of new cases provides a direct, tangible, and actionable metric. Unlike more abstract epidemiological indicators, a case forecast offers a clear outlook that can inform public health planning, resource allocation, and public awareness. It answers the critical, real-world question: "Based on past trends, what can we expect the caseload to be in the near future?
""")

st.header('Model Evaluation')
st.image('Predicted case vs actual case.png')
with st.expander("Model Overview (Click Here)"):
    st.markdown("""
    This model is designed to **predict the smoothed daily new COVID-19 cases in Japan**.
    Its development journey involved several key stages that shaped its current capabilities.
                    """)
with st.expander("2. Model Development Journey (Click Here)"):
    st.subheader("From Complex to Simple, Towards Honest Accuracy")
    st.markdown("""
    Initially, we attempted to use a more complex model, the **Random Forest Regressor**.
    This model showed a very high R^2 (around 99%) on the initial test data. However,
    such high results in real-world data are often a red flag. Our suspicion was confirmed:
    after a stricter cross-validation using **TimeSeriesSplit** (the correct method for time series data),
    the Random Forest model's performance drastically dropped, even showing a negative R^2.
    This indicated "data leakage" or severe overfitting, where the model was memorizing training data rather than learning to genuinely predict.

    Facing these challenges, we transitioned to a simpler and more robust model: **Ridge Regression**.
    This approach proved to be more stable and yielded more reliable evaluations.
    """)
    st.subheader("How the Model Works (Ridge Regression)")
    st.markdown("""
    The **Ridge Regression** model is a relatively simple yet effective type of linear model.
    It works by finding linear relationships between the given features and the number of new cases to be predicted.
    The simplicity of this model helps prevent overfitting, making it better at generating predictions on unseen data.
    """)
with st.expander("3. Features Used by the Model"):
    st.subheader("Key for Time Series Prediction")
    st.markdown("""
    This model heavily relies on **historical data of new cases itself**.
    We used the top 5 most important features from previous analyses (which primarily consist of lagged/historical data of new cases).
    These features provide the model with "memory" about past case trends, which are the most powerful predictors for time series data.
    Additionally, the model incorporates other relevant features (e.g., from the top features identified by the earlier Random Forest model, such as total boosters, day of year, stringency index, etc., that did not cause data leakage).
    """)

with st.expander("4. Model Performance Evaluation"):
    st.subheader("R^2 Score: 0.801")
    st.markdown("""
    This is a very good score! It means the model is capable of explaining approximately **80.1%** of the entire variation (fluctuations) in the smoothed new case count.
    For complex real-world time series predictions, an R^2 of 0.801 indicates **strong and reliable predictive capability**.
    """)
    st.subheader("Mean Squared Error (MSE): 769,853,546")
    st.markdown("""
    Although this number appears large, it's normal because the scale of the new case count itself is also large.
    For easier understanding, we can look at the **Root Mean Squared Error (RMSE)**, which is the square root of MSE.
    $\\text{RMSE} = \\sqrt{769,853,546} \\approx \\textbf{27,746 cases}$
    This means, on average, the model's predictions deviate by about **27,746 cases** from the actual values.
    If the daily smoothed case count can reach hundreds of thousands, a deviation of 27 thousand cases is considered a very good level of accuracy.
    """)

with st.expander("5. Visualization of Prediction Results"):
    st.subheader("Actual vs. Predicted Cases Scatter Plot")
    st.markdown("""
    The scatter plot **'Predicted case vs. Actual case Comparison'** visually confirms the model's performance.
    The blue dots (predictions) are tightly clustered around the red dashed line (the perfect prediction line).
    This demonstrates that the model can predict new case counts with high accuracy across various ranges of values.
    """)



# Define features and target variable
target_col = 'new_cases_smoothed'
x	= df[['new_cases_smoothed_lag_7',
 'new_cases_smoothed_roll_mean_14',
 'hosp_patients',
 'reproduction_rate',
 'new_cases_smoothed_roll_std_14']]
y = df[target_col]

# Split the date into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dfc = df.copy(deep=True)
last_30_days = dfc.tail(30)

# iterative prediction function
def create_prediction():
  

    # last row of dataframe as starting point for prediction
    last_row = dfc.iloc[[-1]].copy()

    # List to save prediction results and dates
    future_predictions = []
    future_dates = []

    # iterate prediction of new cases for next 7 days
    for i in range(7):
        # features for prediction (using the last row)
        features_for_pred = last_row[x.columns]
        
        # predict the next day new cases
        next_day_pred = new_cases_prediction_model.predict(features_for_pred)[0]
        
        # save the prediction result and date in previous lists
        future_predictions.append(next_day_pred)
        last_date = last_row['date'].iloc[0]
        next_date = last_date + pd.Timedelta(days=1)
        future_dates.append(next_date)
        
        # create a new row for next day prediction using the last row (using predicted value)
        new_row = last_row.copy()
        new_row['date'] = next_date
        new_row['new_cases_smoothed'] = next_day_pred 
        
        # update lagged features adjusting the lag values with the new row (lag 7 for today is lag 6 for yesterday, etc.)
        for lag in [7, 14, 21]:
            if f'new_cases_smoothed_lag_{lag}' in new_row.columns:
                prev_lag_col = f'new_cases_smoothed_lag_{lag}' if lag > 1 else 'new_cases_smoothed'
                new_row[f'new_cases_smoothed_lag_{lag-1}'] = last_row[prev_lag_col].iloc[0]

        # Update rolling features with recalculated values
        temp_series = pd.concat([dfc['new_cases_smoothed'], pd.Series(future_predictions)])
        new_rolling_mean = temp_series.rolling(window=14).mean().iloc[-1]
        new_row['new_cases_smoothed_roll_mean_14'] = new_rolling_mean
        
        # use new row as the last row for the new itterationn
        last_row = new_row

        # make a new dataframe for the forecasted results
        forecast_df = pd.DataFrame({'date': future_dates, 'predicted_cases': future_predictions})
        
    return forecast_df

forecast_df = create_prediction()
forecast_df = create_prediction()

# Create a figure and axes for the plot
fig3, ax3 = plt.subplots(figsize=(12, 7))

# Plot historical data
ax3.plot(
    last_30_days['date'], 
    last_30_days['new_cases_smoothed'], 
    label='Historical data (last 30 days)', 
    color='cornflowerblue'
)

# Plot forecasted data
ax3.plot(
    forecast_df['date'], 
    forecast_df['predicted_cases'], 
    label='7 days Prediction', 
    color='red', 
    marker='o', 
    linestyle='--'
)

# Set plot title and labels
ax3.set_title('New Covid Case Prediction in Japan', fontsize=16)
ax3.set_ylabel('Average New Cases (7-Days)')
ax3.set_xlabel('Date')

# Add legend and grid
ax3.legend()
ax3.grid(True)

# Rotate x-axis tick labels for better readability
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

# Display the Matplotlib plot in Streamlit
st.pyplot(fig3, use_container_width=True)
with st.expander("Forcast Process (Click Here)"):
    st.markdown("""
    ## How We Performed the COVID-19 Case Forecast?
    Our goal was to predict new COVID-19 cases for 7 days into the future using a Ridge Regression model. We employed a recursive (iterative) forecasting method, meaning we predicted one day at a time, using that prediction as input for the next.

    **Here's the core process**:

    1. **Starting Point**: We began with the very last known actual data point from our historical dataset.

    Then Daily Prediction Loop (for 7 days):

    2. **Feature Preparation**: For each day we wanted to predict, we prepared the input features. The most critical part here was updating the lagged new case counts (e.g., cases from 1, 7, 14, 21 days ago). We dynamically updated these lags: the predicted case count for the current day became the '1-day lag' for the next day's prediction, and older lags shifted accordingly.

    3. **Other Features**: For other features (like day_of_year, total_boosters, stringency_index), day_of_year was incremented, while others were assumed to remain constant based on the last known values (or extrapolated if their trend was predictable).

    4. **Prediction**: This prepared set of features was fed into our Ridge Regression model to generate a forecast for the next single day.

    5. **Update State**: The newly predicted case count for that day then replaced the 'actual' count in our working dataset for the next iteration, effectively becoming part of the "history" for the subsequent prediction.

    6. **Outcome**: This iterative process allowed us to build a 7-day forward-looking forecast, showing the projected trend of new COVID-19 cases based on our model's understanding of the underlying dynamics
        """)
st.markdown("---")
