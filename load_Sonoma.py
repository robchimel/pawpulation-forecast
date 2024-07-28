import pandas as pd
import os

def load_Sonoma(params, data = 'Animal_Shelter_Intake_and_Outcome_20240517.csv', API=False):
    '''
    prepare sonoma data to be merged with other datasets. clean and do feature engineering
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
    if isinstance(data, pd.DataFrame):
            # 'intake_total':, 
        colmap = {
            'name':'Name', 'type':'Type', 'breed':'Breed', 'color':'Color', 'sex':'Sex', 
            'size':'Size', 'impound_number':'Impound Number',
            'kennel_number':'Kennel Number', 'id':'Animal ID', 'intake_date':'Intake Date', 
            'days_in_shelter':'Days in Shelter', 
            'intake_condition':'Intake Condition', 'intake_jurisdiction':'Intake Jurisdiction',
            'date_of_birth':'Date Of Birth', 
            'outcome_date':'Outcome Date', 'outcome_type':'Outcome Type',
            'outcome_subtype':'Outcome Subtype', 'outcome_condition':'Outcome Condition', 
            'outcome_jurisdiction':'Outcome Jurisdiction',
            'zip_code':'Outcome Zip Code', 'location':'Location',
            'intake_subtype':'Intake_Subtype','intake_type':'Intake_Type'
        }
        df = data.rename(columns=colmap)
        df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    else:
        if os.path.isfile(data)==False:
            data = os.path.join(os.getcwd(), 'Data', data)
        df =  pd.read_csv(data, dtype=dtype_dict)
    df['Date Of Birth'].fillna('01/01/1900', inplace=True)
    # Convert 'Date Of Birth' to datetime after reading the CSV df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Intake Date'] = pd.to_datetime(df['Intake Date'], errors='coerce')
    if 'Outcome Date' in df.columns:
        df['Outcome Date'] = ''
    df = clean_df(df, params, API)
    df = feature_eng(df)

    return df

def feature_eng(df):
    '''this is where we add more columns'''

    # identify return animals, add new col for this
    if 'Animal_ID' in df.columns:
        df['Multiple_Visit_Count'] = df.groupby('Animal_ID')['Animal_ID'].transform('count')
    else:
        df['Multiple_Visit_Count'] = -1

    # calculate age at time of adoption

    df = df[df.Date_Of_Birth!='Unknown']
    df = df[df.Intake_Date!='Unknown']
    df['Intake_Date'] = pd.to_datetime(df['Intake_Date'], errors='coerce')
    df['Date_Of_Birth'] = pd.to_datetime(df['Date_Of_Birth'], errors='coerce')
    df['Age_inDays_at_Income'] = (df['Intake_Date'] - df['Date_Of_Birth']).dt.days

    # Create age groups
    bins = [0, 1, 3, 10, 40, 1000]  # Example bins for age groups
    bins_days = [day*365 for day in bins]
    labels = ['Puppy/Kitten', 'Young', 'Adult', 'Senior', 'Unknown']
    df['Age_Group'] = pd.cut(df['Age_inDays_at_Income'], bins=bins_days, labels=labels)
    # df['Age_Group'].fillna('Unknown', inplace=True)

    # Example of feature interaction
    df['Is_Aggressive'] = int(-1)

    # 1 if animal came to shelter with name, 0 f shelter named animal
    df['Has_Name'] = df['Name'].apply(lambda x: 0 if ('*' in str(x) or 'Unknown' == str(x)) else 1)

    # 1 if animal is fixed, else 0
    df['Is_Fixed'] = df.Sex.apply(lambda x: 1 if 'NEUTERED' in str(x) or 'SPAYED' in str(x) else 0)
    # animal gender ignoring fix status
    df['Sex'] = df.Sex.apply(lambda x: 'MALE' if 'NEUTERED' in str(x) else str(x))
    df['Sex'] = df.Sex.apply(lambda x: 'FEMALE' if 'SPAYED' in str(x) else str(x))

    # are multiple breeds listed?
    df['Is_Mixed_Breed'] = df.Breed.apply(lambda x: 1 if '/' in str(x) or '&' in str(x) or 'MIX' in str(x) else 0)

    # are multiple breeds listed?
    df['Is_Multicolor'] = df.Color.apply(lambda x: 1 if '/' in str(x) else 0)

    return df

def clean_df(df, params, API):
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
    if API==False:
        df = df[~df.Outcome_Date.isnull()]
    else:
        if 'Outcome Date' in df.columns:
            df.loc[df.Outcome_Date.isnull(), 'Days_in_Shelter'] = -1
        else:
            df['Days_in_Shelter'] = -1
            df['Outcome_Date'] = 'Unknown'

    df = df[~df.Intake_Date.isnull()]
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Drop rows where 'Animal ID' is missing as it is a critical identifier
    if 'Animal_ID' in df.columns:
        df.dropna(subset=['Animal_ID'], inplace=True)
    if 'intake_total' in df.columns:
        df.drop(columns=['intake_total'], inplace=True)
    if 'Count' in df.columns:
        df.drop(columns=['Count'], inplace=True)
    # update TORTIE to Tortoiseshell
    df.loc[df.Color=='TORTIE', 'Color'] = 'Tortoiseshell'.upper()
    df.loc[df.Intake_Type=='OS APPT'] = 'Unknown'
    # place nan token for remaining columns
    for col in df.columns:
        null_count = df[col][df[col].isnull()].shape[0]
        if null_count!= 0 and params['na_data'] != False:
            if params['na_data'].lower() == 'fill':
                if (df[col].dtype != float or df[col].dtype != int):
                    print(f"replace null values in {col} with 'Unknown'")
                    df[col].fillna('Unknown', inplace=True)
                else:
                    print(f"replace null values in {col} with '-1'")
                    df[col].fillna(int(-1), inplace=True)

            elif params['na_data'].lower() == 'drop':
                print(f"drop null values in {col}")
                df.dropna(subset=[col], inplace=True)

    if params['drop_outlier_days'] != False:
        df = df[df.Days_in_Shelter < int(params['drop_outlier_days'])]

    return df

if __name__ == '__main__':

    params = {
            'na_data': 'fill',
            'drop_outlier_days': 300,
            'embed':True,
            'num_buckets':3,
            'sample_dict':
                {
                'stratify_col':'Type',
                'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
                }
            }
    df = load_Sonoma(params)
