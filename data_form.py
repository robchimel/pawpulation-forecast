import numpy as np
import pandas as pd
import streamlit as st
from dashboard_utils import MODEL_COLS, TIME_BIN_DICT
import os
import pickle
from utils import *

FORM_DATA = {}
prediction_text = ''
st.markdown("""
# Intake Data Entry
Please input the requested intake data below. When finished, click the
"Generate Prediction" to see the predicted length-of-stay.
""")

with st.form("intake_form"):
    FORM_DATA['Name'] = st.text_input("What is their name?", None)

    FORM_DATA['Type'] = st.selectbox("What is their type?",("Dog", "Cat"),None)

    FORM_DATA['Breed'] = st.text_input("What is their breed?", None)

    FORM_DATA['Color'] = st.text_input("What is their color?", None)

    FORM_DATA['Sex'] = st.selectbox("What is their sex?", ("Female", "Male", "Unknown"), index=None)

    FORM_DATA['Size'] = st.selectbox("What is their size?", ("Small", "Medium", "Kittn", "Large", "Toy", "Puppy", "X-LRG", "Unknown"), index=None)

    FORM_DATA['Date of Birth'] = st.date_input("When is their birthday?", value=None)

    FORM_DATA['Kennel Number'] = st.text_input("What is their kennel number?", 'DA21')

    FORM_DATA['Intake Date'] = st.date_input("When did they go through intake processing?", value=None)

    FORM_DATA['Days in Shelter'] = st.date_input("How many days have they been in the shelter?", value=35)

    FORM_DATA['Intake Type'] = st.selectbox("What is their intake type?",
    ("Stray", "Owner Surrender", "Confiscate", "Quarantine", "Adoption Return", "Transfer", "Born Here", "Unknown"), index=None)

    FORM_DATA['Intake Subtype'] = st.selectbox("What is their intake subtype?",
    ("Field", "Over the counter", "Comm cat", "Fld_arrest", "phone", "vet_hosp", "fld_stray", "fld_hosptl", "priv_shelt", "born_here",
     "fld_coronr", "fld_cruel", "mun_shelt", "field_return to owner", "fld_evict", "field_os", "fld_aband", "email", "over the counter_os",
     "mom stray", "over the counter_return to owner", "over the counter_owned", "rescue_grp", "fld_invest", "over the counter_arrest", "fld_owned",
     "fld_livstk", "over the counter_coronr", "over the counter_evict", "mom os", "over the counter_emerge", "Unknown"),index=None)

    FORM_DATA['Intake Condition'] = st.selectbox(
    "What is their condition?",
    ("Healthy", "Treatable/Rehab", "Untreatable", "Treatable/Manageable","Unknown"),index=None)

    FORM_DATA['Intake Jurisdiction'] = st.selectbox(
    "What is your shelter's jurisdiction?",
    ("Santa Rosa", "County", "Windsor", "Out of County", "Rohnert Park", "Healdsburg", "Sonoma", "Petaluma", "Cloverdale", "sebastopol",
     "tribal resv", "Cotati", "Unknown"),index=None)

    submitted = st.form_submit_button("Generate Prediction")
st.form_submit_button()
if submitted:
    # Run data through data pipeline
    df = pd.DataFrame(FORM_DATA, index=[0])

    if df.shape[0] == 0:
        st.error("No data available for this animal. Please try again.")
    else:
        # Load model and generate prediction
        params = {
        'na_data': 'fill',
        'drop_outlier_days': False,
        'embed':False,
        'buckets':[-1,3,14,30,100,99999999],
        'sample_dict':
            {
            'stratify_col':'Type',
            'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
            }
        }
        df = load_df(params, data=df, split_data=False)
        if os.path.isfile(os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')
        if os.path.isfile(os.path.join(os.getcwd(), 'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.getcwd(), 'XGBpipeline.pkl')
        with open(pipeline_path, 'rb') as file:
            XGBpipeline = pickle.load(file)
        # Predict on the test data
        _, features, _, _, _ = sklearn_pipeline(df, df)
        prediction_text = XGBpipeline.predict(features)
        
st.markdown(f"The animal is predicted to stay for {prediction_text}.")
