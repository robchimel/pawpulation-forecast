from datetime import datetime, timedelta
from sodapy import Socrata
import pandas as pd
import altair as alt
import streamlit as st
import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(2, os.path.dirname(os.getcwd()))
from utils import *
###############################################################################
# Constants
###############################################################################

MODEL_COLS = range(1-6)  # TODO: populate with actual model columns

TIME_BIN_DICT = {
    0: "0-3 Days",
    1: "3-14 Days",
    2: "14-30 Days",
    3: "30-100 Days",
    4: "100+ Days",
}


###############################################################################
# Functions
###############################################################################
@st.cache_data
def get_data_from_API(start_date, end_date):
    """
    Retrieves data from the Sonoma County API for animals with an intake date
    within the `start_date` and `end_date` period

    Args:
        start_date (datetime.datetime): The starting intake date
        end_date (datetime.datetime): The ending intake date

    Returns:
        results_df (pandas.DataFrame): A DataFrame containing the API query results
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    client = Socrata("data.sonomacounty.ca.gov", None)
    results = client.get(
        "924a-vesw",
        where=f"intake_date between '{start_date_str}' and '{end_date_str}'",
    )
    results_df = pd.DataFrame.from_records(results)
    if len(results_df) == 0:
        return pd.DataFrame()
    else:
        params = {
            'na_data': 'fill',
            'drop_outlier_days': False,
            'embed':True,
            'buckets':[-1,3,14,30,100,99999999],
            'sample_dict':
                {
                'stratify_col':'Type',
                'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
                }
            }
        results_df = load_df(params, data=results_df, split_data=False)
        print('\ndata pipeline complete\n')
        return results_df

def plotting_df_from_pred_df(pred_df):
    plot_df = pd.DataFrame(pred_df[["Animal_ID", "Intake_Date", "Days_in_Shelter_Prediction", "Days_in_Shelter_Label_and_Prediction", "Prediction"]])
    plot_df["LOS_Barlength"] = plot_df["Days_in_Shelter_Prediction"] + 1
    plot_df["LOS_Text"] = plot_df["Days_in_Shelter_Prediction"].apply(lambda x: TIME_BIN_DICT[x])
    plot_df["Calendar_Legend_Label"] = plot_df.Prediction.apply(lambda x: {False: "Actual Stay", True: "Predicted Stay"}[x])

    plot_df["LOS_Days"] = plot_df["Days_in_Shelter_Label_and_Prediction"]
    plot_df.loc[plot_df.Prediction.eq(True) & plot_df.Days_in_Shelter_Prediction.eq(0), "LOS_Days"] = 3
    plot_df.loc[plot_df.Prediction.eq(True) & plot_df.Days_in_Shelter_Prediction.eq(1), "LOS_Days"] = 14
    plot_df.loc[plot_df.Prediction.eq(True) & plot_df.Days_in_Shelter_Prediction.eq(2), "LOS_Days"] = 30
    plot_df.loc[plot_df.Prediction.eq(True) & plot_df.Days_in_Shelter_Prediction.eq(3), "LOS_Days"] = 100
    plot_df.loc[plot_df.Prediction.eq(True) & plot_df.Days_in_Shelter_Prediction.eq(4), "LOS_Days"] = 150
    plot_df.loc[plot_df.Prediction.eq(False) & plot_df.LOS_Days.eq(0), "LOS_Days"] = 1 # Make 1 day minimum to display

    plot_df["Outtake_Date"] = plot_df.apply(lambda x: x.Intake_Date + timedelta(days=x.LOS_Days), axis=1)

    return plot_df

def plot_predictions(plot_df, sort_field, order):
    chart = alt.Chart(plot_df[plot_df.Prediction.eq(True)]).mark_bar(orient="horizontal").encode(
        x=alt.X("LOS_Barlength", title="Predicted Bin", axis=alt.Axis(ticks=False, labels=False)),
        y=alt.Y("Animal_ID", title="Animal ID", sort=alt.EncodingSortField(field=sort_field, op="max", order=order)),
        color=alt.Color("LOS_Text", title="Duration", scale=alt.Scale(domain=list(TIME_BIN_DICT.values()))),
    ).properties(title="Length-of-Stay Predictions", width=500)

    return chart

def plot_calendar_view(plot_df, sort_field, order):
    # Brush code obtained from https://stackoverflow.com/a/78118916
    brush = alt.selection_interval(encodings=['x'], bind='scales')

    #chart = alt.Chart(plot_df).mark_bar(orient="horizontal").encode(
    chart = alt.Chart(plot_df[plot_df.Prediction.eq(True)]).mark_bar(orient="horizontal").encode(
        x=alt.X("Intake_Date", title="Intake Date"),
        x2=alt.X2("Outtake_Date", title="Outcome Date"),
        y=alt.Y("Animal_ID",  title="Animal ID", sort=alt.EncodingSortField(field=sort_field, op="max", order=order)),
        #color=alt.Color("Calendar_Legend_Label", title="Type"),
        color=alt.Color("LOS_Text", title="Duration", scale=alt.Scale(domain=list(TIME_BIN_DICT.values()))),
    ).add_params(brush).properties(title="Length-of-Stay Overview", width=500)

    return chart

###############################################################################
# Classes
###############################################################################
if __name__ == '__main__':
    from datetime import datetime, timedelta
    client = Socrata("data.sonomacounty.ca.gov", None)

    end_date_str = datetime.now().strftime("%Y-%m-%d")
    start_date = datetime.now() - timedelta(days = 50)
    start_date_str = start_date.strftime("%Y-%m-%d")
    results = client.get(
        "924a-vesw",
        where=f"intake_date between '{start_date_str}' and '{end_date_str}'",
    )
    results_df = pd.DataFrame.from_records(results)
    params = {
        'na_data': 'fill',
        'drop_outlier_days': False,
        'embed':True,
        'buckets':[-1,3,14,30,100,99999999],
        'sample_dict':
            {
            'stratify_col':'Type',
            'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
            }
        }
    results_df = load_df(params, data=results_df, split_data=False)
