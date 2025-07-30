import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nbformat
# this page is for saving functions


# --- CUSTOM CSS STYLING ---
# This ccs code is from ai to help me adjust the theme

def css_load():
    css = """
    <style>
        /* --- GENERAL STYLING --- */
        /* Main app background */
        .stApp {
            background-color: #ffffff; 
        }

        /* --- SIDEBAR STYLING --- */
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #ADD8E6;
        }

        /* Sidebar font */
        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            color: #ffffff; /* Dark grey text for readability on white */
            font-family: 'Verdana', sans-serif; /* A clean, professional font */
            font-size: 30px;
        }

        /* Sidebar header font */
        [data-testid="stSidebar"] .st-emotion-cache-1lcbmx1 {
            color: #ffffff;
            font-family: 'Verdana', sans-serif;
        }


        /* --- KPI CARD STYLING --- */
        /* KPI Metric card style */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 2px solid #FFFFFF;
            border-radius: 15px; /* More rounded corners */
            padding: 20px;
            /* Adding the shadow you wanted */
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }

        /* KPI value text */
        [data-testid="stMetricValue"] {
            color: #333333;
            font-weight: bold;
        }

        /* KPI label text */
        [data-testid="stMetricLabel"] {
            color: #333333;
        }

        /* --- PLOT STYLING --- */
        /* This targets the container that holds your Matplotlib/Seaborn plots */
        [data-testid="stImage"] {
            border-radius: 20px; /* Rounded corners for the plot container */
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* Shadow for the plot container */
            border: 2px solid #FFFFFF; /* A white border to make it pop */
        }

    </style>
    """
    return st.markdown(css, unsafe_allow_html=True)


# --- DATA LOADING (Cached) ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(r"Japan Covid Data\cleaned_data_japan-owid-covid-data.csv")
    return df


# Plot creator functions
def create_lineplot(data_frame, x, y, color=None, title=None, markers= True, line_dash=None, labels=None, title_font_size = 30):
	
	fig = px.line(data_frame=data_frame, x=x, y=y, color=color, markers=markers, line_shape= 'spline', line_dash= line_dash, title=title, labels=labels)
	fig.update_layout(
					plot_bgcolor='#ffffff',
					font_color = '#000000',
					title_font_size = title_font_size,
					xaxis_title = '',
					yaxis_gridcolor='lightgrey',
					yaxis_tickformat = '.2s'
	)

	fig.update_traces(
					mode='lines+markers',
					line_width=3,
					textfont=dict(size=10, color='black')
	)
	return fig

def create_barplot(plot_data=None, x_plot=None, y_plot=None, plot_title='', order=None, color_bar='#CC0000', xlabel_plot='', ylabel_plot=''):
    labels = {x_plot: xlabel_plot, y_plot: ylabel_plot}
    if not xlabel_plot: labels.pop(x_plot, None)
    if not ylabel_plot: labels.pop(y_plot, None)

    fig = px.bar(
        plot_data,
        x=x_plot,
        y=y_plot,
        title=plot_title,
        category_orders={x_plot: order} if x_plot and order else ({y_plot: order} if y_plot and order else None),
        labels=labels,
        color_discrete_sequence=[color_bar])

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=15,
        yaxis_title_font_size=15)

    return fig

