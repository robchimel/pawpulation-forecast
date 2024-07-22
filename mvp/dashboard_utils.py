from sodapy import Socrata
import pandas as pd
import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(2, os.path.dirname(os.getcwd()))
from utils import *
###############################################################################
# Constants
###############################################################################

MODEL_COLS = ["Name", "Type", "Breed", "Color", "Sex", "Size", "Date_Of_Birth", "Kennel_Number", "Intake_Date", "Intake_Type", "Intake_Subtype", "Intake_Condition", "Intake_Jurisdiction"]  # TODO: populate with actual model columns

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
