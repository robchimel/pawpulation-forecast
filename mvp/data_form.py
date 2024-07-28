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
    FORM_DATA[1] = st.text_input("What is their name?", None)

    FORM_DATA[2] = st.selectbox("What is their type?",("Dog", "Cat"))

    FORM_DATA[3] = st.text_input("What is their breed?", None)

    FORM_DATA[4] = st.text_input("What is their color?", None)

    FORM_DATA[5] = st.selectbox("What is their sex?", ("Female", "Male", "Unknown"), index=None)

    FORM_DATA[6] = st.selectbox("What is their size?", ("Small", "Medium", "Kittn", "Large", "Toy", "Puppy", "X-LRG", "Unknown"), index=None)

    FORM_DATA[7] = st.date_input("When is their birthday?", value=None)

    FORM_DATA[8] = st.text_input("What is their kennel number?", None)

    FORM_DATA[9] = st.date_input("When did they go through intake processing?", value=None)

    FORM_DATA[10] = st.selectbox("What is their intake type?",
    ("Stray", "Owner Surrender", "Confiscate", "Quarantine", "Adoption Return", "Transfer", "Born Here", "Unknown"), index=None)

    FORM_DATA[11] = st.selectbox("What is their intake subtype?",
    ("Field", "Over the counter", "Comm cat", "Fld_arrest", "phone", "vet_hosp", "fld_stray", "fld_hosptl", "priv_shelt", "born_here",
     "fld_coronr", "fld_cruel", "mun_shelt", "field_return to owner", "fld_evict", "field_os", "fld_aband", "email", "over the counter_os",
     "mom stray", "over the counter_return to owner", "over the counter_owned", "rescue_grp", "fld_invest", "over the counter_arrest", "fld_owned",
     "fld_livstk", "over the counter_coronr", "over the counter_evict", "mom os", "over the counter_emerge", "Unknown"),index=None)

    FORM_DATA[12] = st.selectbox(
    "What is their condition",
    ("Healthy", "Treatable/Rehab", "Untreatable", "Treatable/Manageable","Unknown"),index=None)

    FORM_DATA[13] = st.selectbox(
    "What is your shelter's jurisdiction?",
    ("Santa Rosa", "County", "Windsor", "Out of County", "Rohnert Park", "Healdsburg", "Sonoma", "Petaluma", "Cloverdale", "sebastopol",
     "tribal resv", "Cotati", "Unknown"),index=None)

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
