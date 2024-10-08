import numpy as np
import pandas as pd
import streamlit as st
from dashboard_utils import MODEL_COLS, TIME_BIN_DICT
import os
import sys
import pickle
from datetime import datetime
from utils import *

FORM_DATA = {}

st.markdown("""
# Intake Data Entry
Please input the requested intake data below. When finished, click the
"Generate Prediction" to see the predicted length-of-stay.
""")

with st.form("intake_form"):
    FORM_DATA[1] = st.text_input("What is their name?", placeholder="FIDO")

    FORM_DATA[2] = st.selectbox("What is their type?",("Dog", "Cat"),placeholder="Dog")

    FORM_DATA[3] = st.text_input("What is their breed?", placeholder="GERM SHEPHERD")

    FORM_DATA[4] = st.text_input("What is their color?", placeholder="TAN/BLACK")

    FORM_DATA[5] = st.selectbox("What is their sex?", ("Female", "Male", "Unknown"), placeholder="Male")

    FORM_DATA[6] = st.selectbox("What is their size?", ("Small", "Medium", "Kittn", "Large", "Toy", "Puppy", "X-LRG", "Unknown"), placeholder="Small")

    FORM_DATA[7] = st.date_input("When is their birthday?", None)

    FORM_DATA[8] = st.text_input("What is their kennel number?", placeholder="DA31")

    FORM_DATA[9] = st.date_input("When did they go through intake processing?", None)

    FORM_DATA[10] = st.selectbox("What is their intake type?",
    ("Stray", "Owner Surrender", "Confiscate", "Quarantine", "Adoption Return", "Transfer", "Born Here", "Unknown"), placeholder="Owner Surrender")

    FORM_DATA[11] = st.selectbox("What is their intake subtype?",
    ("Field", "Over the counter", "Comm cat", "Fld_arrest", "phone", "vet_hosp", "fld_stray", "fld_hosptl", "priv_shelt", "born_here",
     "fld_coronr", "fld_cruel", "mun_shelt", "field_return to owner", "fld_evict", "field_os", "fld_aband", "email", "over the counter_os",
     "mom stray", "over the counter_return to owner", "over the counter_owned", "rescue_grp", "fld_invest", "over the counter_arrest", "fld_owned",
     "fld_livstk", "over the counter_coronr", "over the counter_evict", "mom os", "over the counter_emerge", "Unknown"),placeholder="Over the counter")

    FORM_DATA[12] = st.selectbox(
    "What is their condition",
    ("Healthy", "Treatable/Rehab", "Untreatable", "Treatable/Manageable","Unknown"),placeholder="Healthy")

    FORM_DATA[13] = st.selectbox(
    "What is your shelter's jurisdiction?",
    ("Santa Rosa", "County", "Windsor", "Out of County", "Rohnert Park", "Healdsburg", "Sonoma", "Petaluma", "Cloverdale", "sebastopol",
     "tribal resv", "Cotati", "Unknown"),placeholder="County")

    submitted = st.form_submit_button("Generate Prediction")

if submitted:
    # Prediction pipeline
    df = pd.DataFrame(FORM_DATA, index=[0])
    cols = ['Name', 'Type', 'Breed', 'Color', 'Sex', 'Size', 'Date Of Birth',  
        'Kennel Number', 'Intake Date', 'Intake Type', 'Intake Subtype', 
        'Intake Condition', 'Intake Jurisdiction']
    df.columns = cols
    # Run df through data pipeline
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
    results_df = load_df(params, data=df, split_data=False)
    # Load model and generate prediction
    try:
        if os.path.isfile(os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.path.dirname(os.getcwd()),'XGBpipeline.pkl')
        if os.path.isfile(os.path.join(os.getcwd(), 'XGBpipeline.pkl')):
            pipeline_path = os.path.join(os.getcwd(), 'XGBpipeline.pkl')
        with open(pipeline_path, 'rb') as file:
            XGBpipeline = pickle.load(file)
            
        # Predict on the test data
        _, features, _, _, _ = sklearn_pipeline(results_df, results_df)

        results_df['Days_in_Shelter_Prediction'] = XGBpipeline.predict(features)
        prediction_text = results_df.Days_in_Shelter_Prediction.iloc[0]
        predict_los_string = {0:'0 to 3 days', 1:'3 to 14 days', 2:'14 to 30 days', 3:'30 to 100 days', 4:'100+ days'}
        st.markdown(f"The animal is predicted to stay for {predict_los_string[prediction_text]}.")
    except Exception as e:
        st.markdown(f"Error with user input, prediction not returned.\n {e}")
