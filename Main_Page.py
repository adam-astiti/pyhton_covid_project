import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nbformat


# --- PAGE CONFIG ---
# Sets the title of the browser tab, adds an icon, and uses a wide layout for more space.
st.set_page_config(
    page_title="Japan Covid Data Analysis",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import css_load
from utils import load_and_clean_data
from utils import create_lineplot

css_load()
df = load_and_clean_data()

# Create variable for chart
test_case_and_death = df.groupby(['year', 'month'])[['total_cases', 'total_deaths', 'total_tests']].max().reset_index()
test_case_and_death['date'] = test_case_and_death['year'].astype(str) + '-' + test_case_and_death['month'].astype(str)
test_case_and_death_melt	= test_case_and_death.melt(id_vars=['year', 'month', 'date'], value_vars=['total_cases', 'total_deaths', 'total_tests'], value_name='value')
test_case_and_death_melt['variable'] = test_case_and_death_melt['variable'].map({'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths', 'total_tests': 'Total Tests'})


df['IFR'] = df['IFR'] * 10
ifr_vaccine_st = df.groupby(['year', 'month'])[['IFR', 'perc_vaccinated_one_dose', 'stringency_index']].mean().reset_index().sample(25).sort_index(ascending=True)
ifr_vaccine_st['date'] = ifr_vaccine_st['year'].astype(str) + '-' + ifr_vaccine_st['month'].astype(str)
ifr_vaccine_st_melt = ifr_vaccine_st.melt(id_vars=['year', 'month','date'], value_vars=['IFR', 'perc_vaccinated_one_dose', 'stringency_index'], value_name='value')
ifr_vaccine_st_melt['variable'] = ifr_vaccine_st_melt['variable'].map({'IFR': 'Infection Fatality Rate (%)', 'stringency_index': 'Government Restriction Level (%)', 'perc_vaccinated_one_dose': 'Population with 1st Vaccine Dose (%)'})


# --- MAIN PAGE LAYOUT ---
# Layout and title.
st.image(r"copid.jpg", use_container_width=True)
st.title("Understanding COVID-19 Trends in Japan")
st.markdown("---") 
# A subtle separator line


# --- SIDEBAR FILTERS ---

with st.sidebar:
    selected_chart = st.sidebar.radio('Select Chart', ('Overall Trends', 'Total Tests', 'Total Deaths', 'Total Cases'))
    st.divider()
    st.image('D:\Project\pyhton_covid_project\streamlit-logo-primary-colormark-darktext.png', use_container_width=False)
    st.write("""### Project by: **Adam Astiti**""")
    st.link_button('Linkedin', 'https://www.linkedin.com/in/adam-astiti-a3787312a/', icon='💼')
    st.link_button('Github', 'https://github.com/adam-astiti', icon='👨‍💻')




# Create columns for important information
total_tests = test_case_and_death['total_tests'].max()
total_cases = test_case_and_death['total_cases'].max()
total_deaths = test_case_and_death['total_deaths'].max()

# divide into 3 column
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Test", f"{total_tests:,.2f}")

with col2:
    st.metric("Total Cases", f"{total_cases:,.2f}")
    
with col3:
    st.metric("Total Deaths", f"{total_deaths:,.2f}")
st.markdown("---")

# Plotly chart and explanation with expander
if selected_chart == 'Overall Trends':
    fig=	create_lineplot(title='Overall Trends: Total Cases, Deaths, and Testing Over Time',
                            data_frame=test_case_and_death_melt, 
                            x='date', 
                            y='value',
                            color= 'variable',
                            markers= True,
                            line_dash=None, 
                            labels={'value':'value', 'date':'date'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Chart Explanation (Click Here)"):
        st.write("""
            This graph shows us three things about Covid in Japan:
        1. Tests (Red line): Japan really shot up quickly from 2020 through 2021, and by early 2022, they were doing a huge number of tests, hitting over 50 million. After that, the testing numbers mostly flattened out.
        2. Cases (Dark blue line): Following the rise in testing, the total number of cases (the dark blue line) also started climbing significantly. This makes sense – more tests usually find more cases. The case numbers saw a sharp increase, especially in late 2021 and early 2022, before they, too, leveled off around 33 million by early 2023.
        3. total number of deaths (the light blue line) stayed incredibly low throughout the entire period. Despite millions of tests and tens of millions of cases, the cumulative death count barely moved from the bottom of the chart.
        
        This pattern suggests a few things: either Japan had really effective ways to manage severe illness, or a very high number of cases were mild, or perhaps their vaccination programs (which started later in the timeframe) were very successful in preventing severe outcomes. The way all three lines flatten out after 2022 might also suggest a point where the way these numbers were tracked or reported changed.
        """)

elif selected_chart == 'Total Tests':
    fig =   create_lineplot(title='Covid Total Cases in Japan',
                            data_frame=test_case_and_death, 
                            x='date', 
                            y='total_cases',
                            markers= True,
                            line_dash=None, 
                            labels={'total_test':'Total tests', 'date':'Date'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Chart Explanation (Click Here)"):
        st.write("""
            Graph showing a rapid and sustained increase throughout 2021 and into early 2022, eventually leveling off at approximately 52 million tests by mid-2022.
        """)
elif selected_chart == 'Total Deaths':
    fig =   create_lineplot(title='Covid Total Death in Japan',
                            data_frame=test_case_and_death, 
                            x='date', 
                            y='total_deaths',
                            markers= True,
                            line_dash=None, 
                            labels={'total_deaths':'Total Deaths', 'date':'Date'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Chart Explanation (Click Here)"):
        st.write("""
        This chart details Covid Total Deaths in Japan from 2020 to 2024. 
        Deaths started very low, then saw a gradual increase through 2021 and a more pronounced rise in 2022. There was a sharp surge in early 2023, pushing total deaths to about 75,000. Crucially, the line abruptly flattens from mid-2023, suggesting a change in how cumulative death data was reported, rather than zero new fatalities.
        """)
else:
    fig =   create_lineplot(title='Covid Total Cases in Japan',
                            data_frame=test_case_and_death, 
                            x='date', 
                            y='total_cases',
                            markers= True,
                            line_dash=None, 
                            labels={'total_cases':'Total Cases', 'date':'Date'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Chart Explanation (Click Here)"):
        st.write("""
            Following the initial increase in testing, denoting total cases, also began to rise steadily from late 2020 through 2021, and then sharply accelerated into early 2022. This trend for total cases also plateaued around early 2023, stabilizing at approximately 33 million.
        """)



st.markdown("---")

# Create plotly chart and explanation with expander
fig2 = create_lineplot(title='Japan Fight Against COVID-19: The Link Between Restrictions, Vaccines, and Fatality Rates',
                data_frame=ifr_vaccine_st_melt, 
                x='date', 
				y='value',
                color= 'variable',
                markers= True,
                line_dash=None,
                labels={'date':'Date', 'value': 'Percentage %'},
                title_font_size=20)
st.plotly_chart(fig2, use_container_width=True)
with st.expander("Chart Explanation (Click Here)"):
        st.write("""
            The plot strongly suggests a direct and positive impact of vaccination on COVID-19 outcomes. The surge in vaccine coverage directly coincided with a significant reduction in the Infection Fatality Rate, indicating that vaccines made infections far less deadly.

Furthermore, the data implies that government restrictions were gradually eased as vaccination rates climbed and the fatality rate dropped, suggesting a policy response to increased population immunity and reduced disease severity. This visualization tells a powerful story of how Japan adapted its fight against the virus, with vaccination playing a pivotal role in shifting the pandemic's trajectory towards a less fatal outcome.
        """)
