import pandas as pd
from sklearn.model_selection import train_test_split


def load_df(params, name = 'Animal_Shelter_Intake_and_Outcome_20240517.csv'):
    '''
    this loads the csv, cleans the data and creates new columns

    name is the name of the csv, this should be a direct or relative path to the csv database for Animal_Shelter_Intake_and_Outcome

    params options

    na_data will fill missing data with 'unknown', delete missing data or do nothing
    input options are...
        * 'fill'
        * 'drop'
        * False

    drop_outlier_days removes pets who have a lenght of stay exceeding the value YOU enter
    input options are...
        * False
        * or any integer
    
    sample_dict controls stratified sampling
        * stratify_col: a column name used for stratified sampling... spelling and caps must be exact
        * train_size: a fraction of data you want for the training data
        * validate_size: a fraction of data you want for the validate data
        * test_size: a fraction of data you want for the test data
    '''

    dtype_dict = {
        'Name': 'str',
        'Type': 'str',
        'Breed': 'str',
        'Color': 'str',
        'Sex': 'str',
        'Size': 'str',
        'Date Of Birth': 'str',  
        'Intake Date': 'str', 
        'Outcome Date': 'str', 
        'Days in Shelter': 'int',
        'Impound Number': 'str',
        'Kennel Number': 'str',
        'Animal ID': 'str',
        'Outcome Type': 'str',
        'Outcome Subtype': 'str',
        'Intake Condition': 'str',
        'Outcome Condition': 'str',
        'Intake Jurisdiction': 'str',
        'Outcome Jurisdiction': 'str',
        'Outcome Zip Code': 'str',
        'Location': 'str',
        'Count': 'int'
    }
    df =  pd.read_csv(name, dtype=dtype_dict)
    df['Date Of Birth'].fillna('01/01/1900', inplace=True)
    # Convert 'Date Of Birth' to datetime after reading the CSV df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Intake Date'] = pd.to_datetime(df['Intake Date'], errors='coerce')
    df['Outcome Date'] = pd.to_datetime(df['Outcome Date'], errors='coerce')
    df = clean_df(df, params)
    df = feature_eng(df)

    train_df, validate_df, test_df = train_validate_test_split(df, params)
    return train_df, validate_df, test_df

def feature_eng(df):
    '''this is where we add more columns'''

    # identify return animals, add new col for this
    df['Multiple_Visit_Count'] = df.groupby('Animal_ID')['Animal_ID'].transform('count')

    # calculate age at time of adoption
    df['Age_inDays_at_Outcome'] = (df['Outcome_Date'] - df['Date_Of_Birth']).dt.days

    # Create age groups
    bins = [0, 1, 3, 10, 40, 1000]  # Example bins for age groups
    bins_days = [day*365 for day in bins]
    labels = ['Puppy/Kitten', 'Young', 'Adult', 'Senior', 'Unknown']
    df['Age_Group'] = pd.cut(df['Age_inDays_at_Outcome'], bins=bins_days, labels=labels)
    # df['Age_Group'].fillna('Unknown', inplace=True)

    # Example of feature interaction
    df['Is_Aggressive'] = df['Outcome_Subtype'].apply(lambda x: 1 if 'AGGRESSIVE' in str(x) else 0)

    # 1 if animal came to shelter with name, 0 f shelter named animal
    df['Has_Name'] = df['Name'].apply(lambda x: 0 if '*' in str(x) else 1)

    # 1 if animal is fixed, else 0
    df['Is_Fixed'] = df.Sex.apply(lambda x: 1 if 'NEUTERED' in str(x) or 'SPAYED' in str(x) else 0)
    # animal gender ignoring fix status
    df['Sex'] = df.Sex.apply(lambda x: 'MALE' if 'NEUTERED' in str(x) else str(x))
    df['Sex'] = df.Sex.apply(lambda x: 'FEMALE' if 'SPAYED' in str(x) else str(x))

    return df

def clean_df(df, params):
    '''
    clean_df handles null data, renames columns to be more friendly in python, drops duplicates and more!


    params options
    na_data will fill missing data with 'unknown', delete missing data or do nothing
    input options are...
        * 'fill'
        * 'drop'
        * 'nothing'
    drop_outlier_days removes pets who have a lenght of stay exceeding the value YOU enter
    input options are...
        * False
        * or any integer
    '''

    # remove space from col name, force strings to upper
    col_list = []
    for col in df.columns:
        new_col = col.replace(' ', '_')
        col_list.append(new_col)
        try:
            df[col] = df[col].str.upper()
        except:
            print(f'{col} is NOT A STRING')
    df.columns = col_list

    # drop Other
    df = df[df.Type != 'OTHER']
    # remove animals without an outcome or intake date
    df = df[~df.Outcome_Date.isnull()]
    df = df[~df.Intake_Date.isnull()]
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Drop rows where 'Animal ID' is missing as it is a critical identifier
    df.dropna(subset=['Animal_ID'], inplace=True)
    df.drop(columns=['Count'], inplace=True)

    # place nan token for remaining columns
    for col in df.columns:
        null_count = df[col][df[col].isnull()].shape[0]
        if null_count!= 0 and params['na_data'] != False:
            if params['na_data'].lower() == 'fill':
                print(f"replace null values in {col} with 'Unknown'")
                df[col].fillna('Unknown', inplace=True)

            elif params['na_data'].lower() == 'drop':
                print(f"drop null values in {col}")
                df.dropna(subset=[col], inplace=True)

    if params['drop_outlier_days'] != False:
        df = df[df.Days_in_Shelter < int(params['drop_outlier_days'])]

    return df

def train_validate_test_split(df, params):
    '''split data into train, validate, test'''

    stratify_col = params['sample_dict']['stratify_col']
    train_size = params['sample_dict']['train_size']
    validate_size = params['sample_dict']['validate_size']
    test_size = params['sample_dict']['test_size']
    random_state = 42

    assert train_size + validate_size + test_size == 1, "Train, validate, and test sizes must sum to 1."

    # Split data into train+validate and test
    train_validate, test = train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=random_state)

    # Calculate the proportion of validate
    validate_prop = validate_size / (train_size + validate_size)

    # Split train+validate into train and validate
    train, validate = train_test_split(train_validate, test_size=validate_prop, stratify=train_validate[stratify_col], random_state=random_state)

    return train, validate, test


if __name__ == '__main__':
    '''
    params options

    na_data will fill missing data with 'unknown', delete missing data or do nothing
    input options are...
        * 'fill'
        * 'drop'
        * False

    drop_outlier_days removes pets who have a lenght of stay exceeding the value YOU enter
    input options are...
        * False
        * or any integer
    
    sample_dict controls stratified sampling
        * stratify_col: a column name used for stratified sampling... spelling and caps must be exact
        * train_size: a fraction of data you want for the training data
        * validate_size: a fraction of data you want for the validate data
        * test_size: a fraction of data you want for the test data
    '''

    params = {
            'na_data': 'fill',
            'drop_outlier_days': 300,
            'sample_dict':
                {
                'stratify_col':'Type',
                'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
                }
            }
    train_df, validate_df, test_df = load_df(params)