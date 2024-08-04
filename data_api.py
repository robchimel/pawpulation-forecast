from datetime import datetime
import streamlit as st
import pickle
from dashboard_utils import TIME_BIN_DICT, get_data_from_API, plotting_df_from_pred_df, plot_predictions, plot_calendar_view
import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(2, os.path.dirname(os.getcwd()))
from utils import *

PLOT_FUNCS = {
    "Prediction Summary": plot_predictions,
    "Calendar Overview": plot_calendar_view
}

SORT_FIELD = {
    "Animal ID": "Animal_ID",
    "Length of Stay": "LOS_Days"
}

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

    plot_type = st.radio("Plot Style", PLOT_FUNCS.keys(), index=0)

    sort_field = st.radio("Sort by", SORT_FIELD.keys(), index=0)

    sort_style = st.radio("Order", ["ascending", "descending"], index=0)

    submitted = st.form_submit_button("Generate Report")


if submitted:
    # Run data through data pipeline
    df = get_data_from_API(start_date, end_date)

    if len(df) == 0:
        st.error("No data available for the selected date range. Please try again.")
    else:
        # Load model and generate prediction
        if os.path.isfile(os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')
        if os.path.isfile(os.path.join(os.getcwd(), 'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.getcwd(), 'XGBpipeline.pkl')
        with open(pipeline_path, 'rb') as file:
            XGBpipeline = pickle.load(file)
        # Predict on the test data
        _, features, _, _, _ = sklearn_pipeline(df, df)
        df['Days_in_Shelter_Prediction'] = XGBpipeline.predict(features)
        # Days_in_Shelter_Label_and_Prediction captures Days in Shelter prediction
        # if animal has not been adopted
        # if animal has been adopted (IE: df.Prediction==False) set this column to the actual days in shelter
        df['Days_in_Shelter_Label_and_Prediction'] = df.Days_in_Shelter_Prediction
        # df.loc[df.Prediction==False, 'Days_in_Shelter_Label_and_Prediction'] = df[df.Prediction==False].Days_in_Shelter_Label

        # Export option
        time_stamp = datetime.today().strftime("%Y%m%d")
        st.download_button(
            "Export Report",
            data=df.to_csv(),
            file_name=f"{time_stamp}_pawpulation_forecast.csv",
            mime="text/csv",
        )
        # Plot data
        plot_df = plotting_df_from_pred_df(df)
        chart = PLOT_FUNCS[plot_type](plot_df, SORT_FIELD[sort_field], sort_style)
        chart
