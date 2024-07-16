from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_utils import TIME_BIN_DICT, get_data_from_API

st.markdown("""
# Generate a Report
Please select an intake date range for the report. When finished, click the
"Generate Prediction" to see the predicted length-of-stay for the animals.
""")

with st.form("date_form"):
    start_date, end_date = st.date_input(
        "Intake Date Range",
        (datetime.today(), datetime.today()),
        max_value=datetime.today(),
    )
    submitted = st.form_submit_button("Generate Predictions")


if submitted:
    df = get_data_from_API(start_date, end_date)
    # TODO: Run data through data pipeline - could be empty -> complete!
    # TODO: Load model and generate prediction
    # TODO: Set up plots

    # vvvv Test code vvvv
    # df["Length of Stay"] = np.random.randint(0, 5, len(df))
    # df["Color"] = df["Length of Stay"].apply(lambda x: TIME_BIN_DICT[x])
    # df["Length of Stay"] += 1  # So bars actually show up on plot

    st.bar_chart(data=df, x="Animal_ID", y="Days_in_Shelter_Label", color="Color", horizontal=True)

    st.download_button(
        "Export Report",
        data=df.to_csv(),
        file_name=f"{datetime.today().strftime("%Y%m%d")}_pawpulation_forecast.csv",
        mime="text/csv",
    )
    # ^^^^ Test code ^^^^
