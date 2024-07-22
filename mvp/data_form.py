import numpy as np
import pandas as pd
import streamlit as st
from dashboard_utils import MODEL_COLS, TIME_BIN_DICT
import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(2, os.path.dirname(os.getcwd()))
from utils import *

FORM_DATA = {}

st.markdown("""
# Intake Data Entry
Please input the requested intake data below. When finished, click the
"Generate Prediction" to see the predicted length-of-stay.
""")

with st.form("intake_form"):
    for field in MODEL_COLS:
        FORM_DATA[field] = st.text_input(f"Field #{field}")
        # TODO: Validate data that is entered

    submitted = st.form_submit_button("Generate Prediction")

if submitted:
    # Prediction pipeline
    df = pd.DataFrame(FORM_DATA, index=[0])

    # TODO: Run df through data pipeline
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
    results_df = load_df(params, data=df, split_data=False)
    # TODO: Load model and generate prediction

    # TODO: Format output

    # vvvv Test code vvvv
    prediction = np.random.randint(0, 5)
    prediction_text = TIME_BIN_DICT[prediction]

    st.markdown(f"The animal is predicted to stay for {prediction_text}.")
    # ^^^^ Test code ^^^^
