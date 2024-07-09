import pandas as pd
import numpy as np

def load_Austin(params, 
                intake_name = 'Austin_Animal_Center_Intakes_20240609.csv',
                outcome_name = 'Austin_Animal_Center_Outcomes_20240609.csv'):
    '''
    prepare austin data to be merged with other datasets. clean and do feature engineering
    '''

    intake_df =  prep_intake(pd.read_csv(intake_name))
    outcome_df =  prep_outcome(pd.read_csv(outcome_name))
    df = pd.merge(intake_df, outcome_df, how="left", on=['Name','Type', 'Breed','Color','Animal ID'])
    df = clean_df(df, params)
    df = feature_eng(df)

    return df

def feature_eng(df):
    '''this is where we add more columns'''

    # identify return animals, add new col for this
    df['Multiple_Visit_Count'] = df.groupby('Animal_ID')['Animal_ID'].transform('count')

    # calculate age at time of adoption
    df['Age_inDays_at_Income'] = (df['Intake_Date'] - df['Date_Of_Birth']).dt.days

    # Create age groups
    bins = [0, 1, 3, 10, 40, 1000]  # Example bins for age groups
    bins_days = [day*365 for day in bins]
    labels = ['Puppy/Kitten', 'Young', 'Adult', 'Senior', 'Unknown']
    df['Age_Group'] = pd.cut(df['Age_inDays_at_Income'], bins=bins_days, labels=labels)
    # df['Age_Group'].fillna('Unknown', inplace=True)

    # Example of feature interaction
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

def prep_intake(df):
    #convert to upper case
    df['Name'] = df['Name'].str.upper()
    df['Breed'] = df['Breed'].str.upper()
    df['Color'] = df['Color'].str.upper()
    df['Type'] = df['Animal Type'].str.upper()
    df['Intake Type'] = df['Intake Type'].str.upper()

    #align sex values with sonoma dataset
    df['Sex'] = np.where(df['Sex upon Intake'].str.contains('Neutered'), 'NEUTERED', df['Sex upon Intake'])
    df['Sex'] = np.where(df['Sex'].str.contains('Spayed'), 'SPAYED', df['Sex'])
    df['Sex'] = np.where(df['Sex'] == 'Intact Male', 'MALE', df['Sex'])
    df['Sex'] = np.where(df['Sex'] == 'Intact Female', 'FEMALE', df['Sex'])

    #align condition values with sonoma dataset
    healthy = ['NORMAL', 'AGED','PREGNANT', 'SPACE']
    treatable = ['MED ATTN','SICK', 'INJURED','NURSING','BEHAVIOR','MEDICAL']
    untreatable = ['AGONAL','FERAL','MED URGENT','CONGENITAL','NEUROLOGIC', 'PANLEUK', 'PARVO']
    unknown = ['OTHER','UNKNOWN']

    df['Intake Condition'] = np.where(df['Intake Condition'].str.upper().isin(healthy),'HEALTHY',df['Intake Condition'])
    df['Intake Condition'] = np.where(df['Intake Condition'].str.upper().isin(treatable),'TREATABLE', df['Intake Condition'])
    df['Intake Condition'] = np.where(df['Intake Condition'].str.upper().isin(untreatable),'UNTREATABLE', df['Intake Condition'])
    df['Intake Condition'] = np.where(df['Intake Condition'].str.upper().isin(unknown),'Unknown', df['Intake Condition'])

    #convert to datetime
    df['Intake Date'] = pd.to_datetime(df['DateTime'], errors='coerce')

    #add intake counter to align with sonoma dataset
    df = df[['Name','Type','Breed','Color','Sex','Intake Date','Animal ID','Intake Condition']]
    return df 

def prep_outcome(df):
    #convert to upper case
    df['Name'] = df['Name'].str.upper()
    df['Breed'] = df['Breed'].str.upper()
    df['Color'] = df['Color'].str.upper()
    df['Type'] = df['Animal Type'].str.upper()
    df['Outcome Type'] = df['Outcome Type'].str.upper()

    #align sex values with sonoma dataset
    df['Sex Upon Outcome'] = np.where(df['Sex upon Outcome'].str.contains('Neutered'), 'NEUTERED', df['Sex upon Outcome'])
    df['Sex Upon Outcome'] = np.where(df['Sex Upon Outcome'].str.contains('Spayed'), 'SPAYED', df['Sex Upon Outcome'])
    df['Sex Upon Outcome'] = np.where(df['Sex Upon Outcome'] == 'Intact Male', 'MALE', df['Sex Upon Outcome'])
    df['Sex Upon Outcome'] = np.where(df['Sex Upon Outcome'] == 'Intact Female', 'FEMALE', df['Sex Upon Outcome'])

    #align outcome type with sonoma dataset
    df['Outcome Type'] = np.where(df['Outcome Type'].str.contains('RELOCATE'), 'TRANSFER', df['Outcome Type'])
    df['Outcome Type'] = np.where(df['Outcome Type'].str.contains('RTO'), 'RTOS', df['Outcome Type'])
    df['Outcome Type'] = np.where(df['Outcome Type'].str.contains('MISSING'), 'LOST', df['Outcome Type'])
    df['Outcome Type'] = np.where(df['Outcome Type'].str.contains('EUTHANASIA'), 'EUTHANIZE', df['Outcome Type'])
    df['Outcome Type'] = np.where(df['Outcome Type'].str.contains('DIED'), 'DISPOSAL', df['Outcome Type'])

    #convert to datetime
    df['Outcome Date'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df=df.rename(columns={'Date of Birth':'Date Of Birth'})
    df = df[['Name','Type', 'Breed','Color','Date Of Birth','Animal ID','Outcome Date','Outcome Type','Outcome Subtype']]

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

    col_list = ['Name', 'Type', 'Breed', 'Color', 'Sex',
       'Date Of Birth', 'Intake Date', 'Outcome Date', 
       'Outcome Type', 'Outcome Subtype', 'Intake Condition', 'Animal ID']
    df = df[col_list]
    df['Date Of Birth'].fillna('01/01/1900', inplace=True)
    # Convert 'Date Of Birth' to datetime after reading the CSV df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], errors='coerce')
    df['Intake Date'] = pd.to_datetime(df['Intake Date'], errors='coerce')
    df['Outcome Date'] = pd.to_datetime(df['Outcome Date'], errors='coerce')
    df['Days in Shelter'] = (df['Outcome Date'] - df['Intake Date']).dt.days
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
    df = df[~df.Days_in_Shelter.isnull()]
    df = df[df.Days_in_Shelter>=0]
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Drop rows where 'Animal ID' is missing as it is a critical identifier
    df.dropna(subset=['Animal_ID'], inplace=True)
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

    if params['drop_outlier_days'] != False:
        df = df[df.Days_in_Shelter < int(params['drop_outlier_days'])]

    return df

if __name__ == '__main__':
    params = {
            'na_data': 'fill',
            'drop_outlier_days': 300,
            'embed':True,
            'buckets':[-1,3,14,30,100,99999999],
            'sample_dict':
                {
                'stratify_col':'Type',
                'train_size':0.6, 'validate_size':0.2, 'test_size':0.2
                }
            }
    df = load_Austin(params)
