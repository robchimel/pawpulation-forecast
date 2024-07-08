import pandas as pd
import numpy as np

def load_denver(params, name = 'DAS_Data.csv'):
    '''
    prepare denver data to be merged with other datasets. clean and do feature engineering
    '''

    dtype_dict = {
        'animal_name': 'str', 
        'animal_type': 'str', 
        'primary_breed': 'str', 
        'primary_color': 'str', 
        'sex': 'str',
        'animal_size': 'str', 
        'dob': 'str', 
        'intake_date': 'str', 
        'outcome_date': 'str', 
        'intake_type': 'str',
        'intake_subtype': 'str', 
        'outcome_type': 'str', 
        'outcome_subtype': 'str', 
        'intake_cond': 'str',
        'outcome_cond': 'str', 
        'crossing': 'str'
    }
    df =  pd.read_csv(name, dtype=dtype_dict)

    rename_dict = {'animal_name':'Name', 'animal_type':'Type', 'primary_breed':'Breed', 'primary_color':'Color', 'sex':'Sex',
        'animal_size':'Size', 'dob':'Date Of Birth', 'intake_date':'Intake Date', 'outcome_date':'Outcome Date', 
        'intake_type':'Intake Type', 'intake_subtype':'Intake Subtype',
        'outcome_type':'Outcome Type', 'outcome_subtype':'Outcome Subtype', 'intake_cond':'Intake Condition',
        'outcome_cond':'Outcome Condition'}
    col_list = ['Name', 'Type', 'Breed', 'Color', 'Sex',
       'Size', 'Date Of Birth', 'Intake Date', 'Outcome Date', 'Intake Type',
       'Outcome Type', 'Intake Subtype','Outcome Subtype', 'Intake Condition',
       'Outcome Condition']
    # rename cols like sonoma, did not find a map for 'Outcome Date', 'intake_type'
    df = df.rename(columns=rename_dict)
    # drop 'crossing'
    df = df[col_list]
    # df['Date Of Birth'].fillna('01/01/1900', inplace=True)
    # df['Date Of Birth'] = np.where(df['Date Of Birth']=='00:00.0','01/01/1900',df['Date Of Birth'])
    # df.loc[df['Date Of Birth']=='00:00.0', 'Date Of Birth'] = '01/01/1900'
    # Convert 'Date Of Birth' to datetime after reading the CSV df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    # df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Intake Date'] = pd.to_datetime(df['Intake Date'], errors='coerce')
    df['Outcome Date'] = pd.to_datetime(df['Outcome Date'], errors='coerce')
    df = clean_df(df, params)
    df = feature_eng(df)
    return df

def feature_eng(df):
    '''this is where we add more columns'''
    # calculate age at time of adoption
    # df['Age_inDays_at_Income'] = (df['Intake_Date'] - df['Date_Of_Birth']).dt.days

    # Create age groups
    # bins = [0, 1, 3, 10, 40, 1000]  # Example bins for age groups
    # bins_days = [day*365 for day in bins]
    # labels = ['Puppy/Kitten', 'Young', 'Adult', 'Senior', 'Unknown']
    # df['Age_Group'] = pd.cut(df['Age_inDays_at_Income'], bins=bins_days, labels=labels)
    # df['Age_Group'].fillna('Unknown', inplace=True)

    ## map Denver to Sonoma
    # Intake_Type
    df.loc[df.Intake_Type.isin(['OWNER SUR','RETURN SUR']), 'Intake_Type'] = 'OWNER SURRENDER'
    df.loc[df.Intake_Type=='STRAY TAGS', 'Intake_Type'] = 'STRAY'
    df.loc[df.Intake_Type.isin(['SAFE','PUBLIC SN']), 'Intake_Type'] = 'Unknown'
    df.loc[df.Intake_Type=='ADOPT RET', 'Intake_Type'] = 'ADOPTION RETURN'
    df.loc[df.Intake_Type=='TRANSFR IN', 'Intake_Type'] = 'TRANSFER'
    df.loc[df.Intake_Type=='BORN SHELT', 'Intake_Type'] = 'BORN HERE'
    df.loc[df.Intake_Type=='FOSTER IN', 'Intake_Type'] = 'Unknown'
    # Intake_Condition
    df.loc[df.Intake_Condition.isin(['NORMAL','FEARFUL','UNWEANED', 'NURSING', 'AGED']), 'Intake_Condition'] = 'HEALTHY'
    df.loc[df.Intake_Condition.isin(['FERAL','*AGG HIST*','AGGRESSIVE']), 'Intake_Condition'] = 'UNKNOWN'
    df.loc[df.Intake_Condition=='DEAD', 'Intake_Condition'] = 'UNTREATABLE'
    df.loc[df.Intake_Condition.isin(['SICK','CRUELTY','PREGNANT']), 'Intake_Condition'] = 'TREATABLE/MANAGEABLE'
    df.loc[df.Intake_Condition=='INJURED', 'Intake_Condition'] = 'TREATABLE/REHAB'
    # Intake_Subtype
    df.loc[df.Intake_Subtype.isin(['PFL','BQ','BQ-TEMP', 'S-DV']), 'Intake_Subtype'] = 'UNKNOWN'

    # Example of feature interaction
    df['Is_Aggressive'] = df['Intake_Subtype'].apply(lambda x: 1 if 'BITE' in str(x) else 0)
    df['Is_Aggressive'] = df['Intake_Condition'].apply(lambda x: 1 if ('AGGRESSIVE' in str(x) or '*AGG HIST*' in str(x)) else 0)

    # 1 if animal came to shelter with name, 0 f shelter named animal
    df['Has_Name'] = df['Name'].apply(lambda x: 0 if ('*' in str(x) or 'Unknown' == str(x)) else 1)

    # 1 if animal is fixed, else 0
    df['Is_Fixed'] = df.Sex.apply(lambda x: 1 if ('N' in str(x) or 'S' in str(x)) else 0)
    # animal gender ignoring fix status
    df['Sex'] = df.Sex.apply(lambda x: 'MALE' if ('N' == str(x) or 'M' == str(x)) else str(x))
    df['Sex'] = df.Sex.apply(lambda x: 'FEMALE' if ('S' == str(x) or 'F' == str(x)) else str(x))
    df['Sex'] = df.Sex.apply(lambda x: 'Unknown' if 'U' == str(x) else str(x))

    # are multiple breeds listed?
    # df['Is_Mixed_Breed'] = df.Breed.apply(lambda x: 1 if '/' in str(x) or '&' in str(x) or 'MIX' in str(x) else 0)

    # are multiple breeds listed?
    # df['Is_Multicolor'] = df.Color.apply(lambda x: 1 if '/' in str(x) else 0)

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
    df = df[~df.Type.isin(['OTHER', 'BIRD', 'LIVESTOCK'])]
    # remove animals without an outcome or intake date
    df = df[~df.Outcome_Date.isnull()]
    df = df[~df.Intake_Date.isnull()]
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # No 'Animal ID' or 'Count' col
    # update TORTIE to Tortoiseshell
    df.loc[df.Color=='TORTIE', 'Color'] = 'Tortoiseshell'.upper()
    
    # place nan token for remaining columns
    for col in df.columns:
        null_count = df[col][df[col].isnull()].shape[0]
        if null_count!= 0 and params['na_data'] != False:
            if params['na_data'].lower() == 'fill':
                if (df[col].dtype != float and df[col].dtype != int):
                    print(f"replace null values in {col} with 'Unknown'")
                    df[col].fillna('Unknown', inplace=True)
                else:
                    print(f"replace null values in {col} with '-1'")
                    df[col].fillna(int(-1), inplace=True)

            elif params['na_data'].lower() == 'drop':
                print(f"drop null values in {col}")
                df.dropna(subset=[col], inplace=True)
    # make days in shelter
    df['Days_in_Shelter'] = (df['Outcome_Date'] - df['Intake_Date']).dt.days
    if params['drop_outlier_days'] != False:
        df = df[df.Days_in_Shelter.astype(int) < int(params['drop_outlier_days'])]

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
    
    df = load_denver(params)
