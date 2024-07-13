import streamlit as st

home_page = st.Page("home.py", title="Home", icon="🏚️")
data_input_page = st.Page("data_form.py", title="Enter Intake Data", icon="📝")
data_api_page = st.Page("data_api.py", title="Generate Report", icon="📊")
about_page = st.Page("about.py", title="About/Help", icon="📖")

pg = st.navigation([home_page, data_input_page, data_api_page, about_page])
st.set_page_config(page_title="Pawpulation Forecast", page_icon="🐶")
pg.run()
