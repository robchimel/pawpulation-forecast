import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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
    
    embed creates 50x1 embedding vectors for color and breed.
        download https://nlp.stanford.edu/data/glove.6B.zip, unzip and save in repo
        * True
        * False
    
    sample_dict controls stratified sampling
        * stratify_col: a column name used for stratified sampling... spelling and caps must be exact
        * train_size: a fraction of data you want for the training data
        * validate_size: a fraction of data you want for the validate data
        * test_size: a fraction of data you want for the test data
    
    num_buckets how many buckets to break up length of stay into for model training
        creates new column Days_in_Shelter_Label
        * input is a integer
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

    if params['embed']:
        # download https://nlp.stanford.edu/data/glove.6B.zip, unzip and save in repo
        glove_file_path = 'glove.6B/glove.6B.50d.txt'
        embeddings_index = load_glove_embeddings(glove_file_path)
        df = embed_colors(df, embeddings_index)
        df = embed_breeds(df, embeddings_index)


    num_buckets = params['num_buckets']
    class_labels = [i for i in range(num_buckets)]
    df['Days_in_Shelter_Label'], bin_edges = pd.qcut(df['Days_in_Shelter'], q=num_buckets, labels=class_labels, retbins=True)
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

    # are multiple breeds listed?
    df['Is_Mixed_Breed'] = df.Breed.apply(lambda x: 1 if '/' in str(x) or '&' in str(x) or 'MIX' in str(x) else 0)

    # are multiple breeds listed?
    df['Is_Multicolor'] = df.Color.apply(lambda x: 1 if '/' in str(x) else 0)

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
    # update TORTIE to Tortoiseshell
    df.loc[df.Color=='TORTIE', 'Color'] = 'Tortoiseshell'.upper()
    
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

def load_glove_embeddings(file_path='glove.6B/glove.6B.50d.txt'):
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_word_embedding(word, embeddings_index):
    word = word.lower()
    return embeddings_index.get(word, np.zeros(50))  # 50 is the dimension of the GloVe vectors

def get_mean_color_embedding(color, color_embeddings):
    color = color.replace('/',' ')
    colors = color.split(' ')
    embeddings = [color_embeddings[c] for c in colors if c in color_embeddings]
    return np.mean(embeddings, axis=0)

def get_mean_breed_embedding(breed, breed_embeddings):
    breed = breed.replace('/',' ')
    breed = breed.replace('&',' ')
    breed = breed.replace('   ',' ')
    breed = breed.replace('  ',' ')
    breeds = breed.split(' ')
    embeddings = [breed_embeddings[c] for c in breeds if c in breed_embeddings]
    return np.mean(embeddings, axis=0)

def embed_colors(df, embeddings_index):
    # Extract unique colors and get their embeddings
    unique_colors = df['Color'].str.replace('/', ' ').str.split(' ').explode().unique()
    color_embeddings = {color: get_word_embedding(color, embeddings_index) for color in unique_colors}
    if 'TORTIE' in color_embeddings.keys():
        color_embeddings['TORTIE'] = color_embeddings['Tortoiseshell'.upper()]
    # Apply the function to create embeddings for the 'Color' column
    df['Color_Embedding'] = df['Color'].apply(lambda x: get_mean_color_embedding(x, color_embeddings))
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(np.array(df.Color_Embedding.tolist()))
    # Apply KMeans clustering
    n_clusters = 5  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Color_Embedding_Cluster'] = kmeans.fit_predict(reduced_embeddings)
    return df

def embed_breeds(df, embeddings_index):
    # Extract unique colors and get their embeddings
    unique_breeds = df['Breed'].str.replace('/',' ').str.replace('&',' ').str.split(' ').explode().unique()
    breed_embeddings = {breed: get_word_embedding(breed, embeddings_index) for breed in unique_breeds}
    df['Breed_Embedding'] = df['Breed'].apply(lambda x: get_mean_breed_embedding(x, breed_embeddings))
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(np.array(df.Breed_Embedding.tolist()))
    # Apply KMeans clustering
    n_clusters = 5  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Breed_Embedding_Cluster'] = kmeans.fit_predict(reduced_embeddings)
    return df

def sklearn_pipeline(train_df,validate_df):
    # Define feature columns and target column
    feature_cols = ['Type', 'Sex', 'Size', 'Intake_Type', 'Intake_Subtype',
       'Intake_Condition', 'Multiple_Visit_Count',
       'Age_inDays_at_Outcome', 'Age_Group', 'Is_Aggressive', 'Has_Name',
       'Is_Fixed', 'Is_Mixed_Breed', 'Is_Multicolor', 'Color_Embedding_Cluster',
       'Breed_Embedding_Cluster']
    target_col = 'Days_in_Shelter_Label'

    # Split data into training and testing sets
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = validate_df[feature_cols]
    y_test = validate_df[target_col]

    # Define the column transformer with OneHotEncoder for categorical columns and StandardScaler for numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age_inDays_at_Outcome', 'Multiple_Visit_Count', 'Color_Embedding_Cluster',
                'Breed_Embedding_Cluster']),
            ('cat', OneHotEncoder(), ['Type', 'Sex', 'Size', 'Intake_Type', 'Intake_Subtype',
                'Intake_Condition','Age_Group', 'Is_Aggressive', 'Has_Name',
                'Is_Fixed', 'Is_Mixed_Breed', 'Is_Multicolor'])
        ])

    return preprocessor, X_train, y_train, X_test, y_test


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
    
    embed creates 50x1 embedding vectors for color and breed
        download https://nlp.stanford.edu/data/glove.6B.zip, unzip and save in repo
        * True
        * False
    
    sample_dict controls stratified sampling
        * stratify_col: a column name used for stratified sampling... spelling and caps must be exact
        * train_size: a fraction of data you want for the training data
        * validate_size: a fraction of data you want for the validate data
        * test_size: a fraction of data you want for the test data

    num_buckets how many buckets to break up length of stay into for model training
        creates new column Days_in_Shelter_Label
        * input is a integer
    
    '''

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
    train_df, validate_df, test_df = load_df(params)
