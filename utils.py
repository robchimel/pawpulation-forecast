import pandas as pd


def load_df(name = 'Animal_Shelter_Intake_and_Outcome_20240517.csv'):
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
    df = clean_df(df)
    df = feature_eng(df)
    return df

def feature_eng(df):
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

def clean_df(df):
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
        if null_count!= 0:
            print(f"replace null values in {col} with 'Unknown'")
            df[col].fillna('Unknown', inplace=True)

    return df

if __name__ == '__main__':
    df = load_df()