from sodapy import Socrata
import pandas as pd

###############################################################################
# Constants
###############################################################################

MODEL_COLS = range(1, 6)  # TODO: populate with actual model columns

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
    """ """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    client = Socrata("data.sonomacounty.ca.gov", None)
    results = client.get(
        "924a-vesw",
        where=f"intake_date between '{start_date_str}' and '{end_date_str}'",
    )

    return pd.DataFrame.from_records(results)


###############################################################################
# Classes
###############################################################################
