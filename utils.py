import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from load_Denver import *
from load_Sonoma import *
from load_Austin import *
import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(2, os.path.dirname(os.getcwd()))


def load_df(params, data = False, no_outcome_cols=True, split_data=True, location='all'):
    '''
        data = False if you are not passing a dataframe into load_df and want to use our default data
        no_outcome_cols=True drops excess columns not needed for model training
        split_data = True splits data into Train, Test, Validation
        location controls what locations we pull from. This is only used for data=True (can be all, Sonoma, Austin, Denver)
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

        embed creates 50x1 embedding vectors for color and Subtype
            download https://nlp.stanford.edu/data/glove.6B.zip, unzip and save in repo
            * True
            * False

        sample_dict controls stratified sampling
            * stratify_col: a column name used for stratified sampling... spelling and caps must be exact
            * train_size: a fraction of data you want for the training data
            * validate_size: a fraction of data you want for the validate data
            * test_size: a fraction of data you want for the test data

        buckets what buckets will we split the data to?
            creates new column Days_in_Shelter_Label
            * input is a list of integers
            * please use [-1,3,14,30,100,99999999] as agreed upon based on shelter feedback
    '''
    if isinstance(data, pd.DataFrame):
        df = load_Sonoma(params, data, API=True)
        df['dataset'] = 'API'
        df['Days_in_Shelter'] = df['Days_in_Shelter'].astype('int64')
        # API_data = True
    else:
        if location.title() == 'All':
            df = load_all_data(params) # use multiprocess to speed up data load
            # API_data = False
        elif location.title() == 'Sonoma':
            df = load_Sonoma(params)
            df['dataset'] = 'Sonoma'
        elif location.title() == 'Denver':
            df = load_denver(params)
            df['dataset'] = 'Denver'
        elif location.title() == 'Austin':
            df = load_Austin(params)
            df['dataset'] = 'Austin'

    df['Intake_Year'] = df['Intake_Date'].dt.year
    df['Intake_Month'] = df['Intake_Date'].dt.month
    df['Intake_Day'] = df['Intake_Date'].dt.day
    df['Date_Of_Birth'] = pd.to_datetime(df['Date_Of_Birth'], errors='coerce')
    df['Birth_Year'] = df['Date_Of_Birth'].dt.year
    df['Birth_Month'] = df['Date_Of_Birth'].dt.month
    df['Birth_Day'] = df['Date_Of_Birth'].dt.day
    df = df[df.Intake_Subtype!='S-EVICT']
    # place nan token for remaining columns
    for col in df.columns:
        null_count = df[col][df[col].isnull()].shape[0]
        if null_count!= 0 and params['na_data'] != False:
            if params['na_data'].lower() == 'fill':
                if (df[col].dtype != float and df[col].dtype != int):
                    print(f"replace null values in {col} with 'Unknown'")
                    df[col].fillna('Unknown', inplace=True)
                else:
                    print(f"replace null values in {col} with 'np.nan'")
                    df[col].fillna(int(-1), inplace=True)

            elif params['na_data'].lower() == 'drop':
                print(f"drop null values in {col}")
                df.dropna(subset=[col], inplace=True)

    if params['embed']:
        # download https://nlp.stanford.edu/data/glove.6B.zip, unzip and save in repo
        if os.path.isfile('glove.6B/glove.6B.50d.txt'):
            glove_file_path = 'glove.6B/glove.6B.50d.txt'
        else:
            repo_dir = os.path.dirname(os.getcwd())
            glove_file_path = os.path.join(repo_dir, 'glove.6B', 'glove.6B.50d.txt')

        embeddings_index = load_glove_embeddings(glove_file_path)
        df = embed_colors(df, embeddings_index)
        df = embed_breeds(df, embeddings_index)
        df = embed_subtype(df, embeddings_index)
    else:
        df['Color_Embedding_Cluster'] = -1
        df['Color_Embedding'] = -1
        df['Subtype_Embedding_Cluster'] = -1
        df['Intake_Subtype_Embedding'] = -1
        df['Breed_Embedding_Cluster'] = -1
        df['Breed_Embedding'] = -1

    class_labels = [i for i in range(len(params['buckets'])-1)]
    df['Days_in_Shelter_Label'] = pd.cut(df['Days_in_Shelter'], bins=params['buckets'], labels=class_labels)
    # df['Days_in_Shelter_Label'], bin_edges = pd.qcut(df['Days_in_Shelter'], q=num_buckets, labels=class_labels, retbins=True)
    df['Prediction'] = True
    df.loc[df.Outcome_Date != 'Unknown', 'Prediction'] = False
    if no_outcome_cols==True:
        final_cols = ['Name', 'Type', 'Breed', 'Color', 'Sex', 'Size', 'Date_Of_Birth',
            'Impound_Number', 'Kennel_Number', 'Animal_ID', 'Intake_Date',
            'Days_in_Shelter', 'Intake_Type', 'Intake_Subtype',
            'Intake_Condition', 'Intake_Jurisdiction',
            'Location', 'Multiple_Visit_Count',
            'Age_inDays_at_Income', 'Age_Group', 'Has_Name', 'Is_Fixed',
            'Is_Mixed_Breed', 'Is_Multicolor', 'dataset', 'Is_Aggressive',
            'Color_Embedding', 'Color_Embedding_Cluster', 'Breed_Embedding',
            'Breed_Embedding_Cluster', 'Intake_Subtype_Embedding',
            'Subtype_Embedding_Cluster', 'Days_in_Shelter_Label','Intake_Year',
            'Intake_Month','Intake_Day','Birth_Year','Birth_Month','Birth_Day',
            'Prediction'
            ]
        final_cols = [col for col in final_cols if col in df.columns]

        df = df[final_cols]
        
    
        
    if split_data:
        train_df, validate_df, test_df = train_validate_test_split(df, params)
        return train_df, validate_df, test_df
    else:
        return df

def smap(f):
    return f()
    
def load_all_data(params):
    
    dfSonoma = load_Sonoma(params)
    dfSonoma['dataset'] = 'Sonoma'    
    dfDenver = load_denver(params)
    dfDenver['dataset'] = 'Denver'
    dfAustin = load_Austin(params)
    dfAustin['dataset'] = 'Austin'

    df = pd.concat([dfSonoma,dfDenver,dfAustin], ignore_index=True)
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
    replace_dict = {
        'FLD':'field', 'HOSPTL':'hospital','CORONR':'coroner', 'HOSP':'hospital',
        'ABAND':'abandon', 'SHELT':'shelter', 'LIVSTK':'livestock','EUTH':'euthanasia',
        'HOUSELES':'homeless', 'PRIV':'Unknown','NONCONFIRM':'Unknown', 'SUBSTANC':'Unknown'
    }
    if word in replace_dict.keys():
        word = replace_dict[word]
    word = word.lower()
    return embeddings_index.get(word, np.zeros(50))  # 50 is the dimension of the GloVe vectors

def get_mean_color_embedding(color, color_embeddings):
    color = color.replace('/',' ')
    colors = color.split(' ')
    embeddings = [color_embeddings[c] for c in colors if c in color_embeddings]
    return np.mean(embeddings, axis=0)

def get_mean_breed_embedding(Subtype, breed_embeddings):
    Subtype = Subtype.replace('/',' ')
    Subtype = Subtype.replace('&',' ')
    Subtype = Subtype.replace('   ',' ')
    Subtype = Subtype.replace('  ',' ')
    breeds = Subtype.split(' ')
    embeddings = [breed_embeddings[c] for c in breeds if c in breed_embeddings]
    return np.mean(embeddings, axis=0)

def get_mean_subtype_embedding(Subtype, Subtype_embeddings):
    Subtype = Subtype.replace('-',' ')
    Subtype = Subtype.replace('_',' ')
    Subtype = Subtype.replace('   ',' ')
    Subtype = Subtype.replace('  ',' ')
    Subtypes = Subtype.split(' ')
    embeddings = [Subtype_embeddings[c] for c in Subtypes if c in Subtype_embeddings]
    return np.mean(embeddings, axis=0)

def embed_colors(df, embeddings_index):
    # Extract unique colors and get their embeddings
    unique_colors = df['Color'].str.replace('/', ' ').str.split(' ').explode().unique()
    color_embeddings = {color: get_word_embedding(color, embeddings_index) for color in unique_colors}
    try:
        if 'TORTIE' in color_embeddings.keys():
            color_embeddings['TORTIE'] = color_embeddings['Tortoiseshell'.upper()]
    except:
        print('failed to rename TORTIE color_embedding')
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

def embed_subtype(df, embeddings_index):
    # Extract unique colors and get their embeddings
    unique_Subtype = df['Intake_Subtype'].str.replace('-',' ').str.replace('_',' ').str.split(' ').explode().unique()
    Subtype_embeddings = {Subtype: get_word_embedding(Subtype, embeddings_index) for Subtype in unique_Subtype}
    df['Intake_Subtype_Embedding'] = df['Intake_Subtype'].apply(lambda x: get_mean_subtype_embedding(x, Subtype_embeddings))
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(np.array(df.Intake_Subtype_Embedding.tolist()))
    # Apply KMeans clustering
    n_clusters = 5  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Subtype_Embedding_Cluster'] = kmeans.fit_predict(reduced_embeddings)
    return df

def sklearn_pipeline(train_df,validate_df):
    # Define feature columns and target column
    
    # dropped 'Breed', 'Color' these had issues with pipeline
    feature_cols = [
                    'Kennel_Number', 'Intake_Jurisdiction', 
                    'Type', 'Sex', 'Size', 'Intake_Type', 'Intake_Subtype',
                    'Intake_Condition', 'Multiple_Visit_Count',
                    'Age_inDays_at_Income', 'Age_Group', 'Is_Aggressive', 'Has_Name',
                    'Is_Fixed', 'Is_Mixed_Breed', 'Is_Multicolor', 'Color_Embedding_Cluster',
                    'Breed_Embedding_Cluster', 
                    'Intake_Year','Intake_Month','Intake_Day','Birth_Year','Birth_Month','Birth_Day'
                ]
    num_cols = [
                'Age_inDays_at_Income', 'Multiple_Visit_Count', 'Color_Embedding_Cluster',
                'Breed_Embedding_Cluster', 'Intake_Year','Intake_Month',
                'Intake_Day','Birth_Year','Birth_Year','Birth_Day'
                ]
    cat_cols = [
                'Kennel_Number', 'Intake_Jurisdiction'
                'Type', 'Sex', 'Size', 'Intake_Type', 'Intake_Subtype',
                'Intake_Condition','Age_Group', 'Is_Aggressive', 'Has_Name',
                'Is_Fixed', 'Is_Mixed_Breed', 'Is_Multicolor'
                ]
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    num_cols = [col for col in num_cols if col in feature_cols]
    cat_cols = [col for col in cat_cols if col in feature_cols]
    target_col = 'Days_in_Shelter_Label'

    # Split data into training and testing sets
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = validate_df[feature_cols]
    y_test = validate_df[target_col]

    # Define the column transformer with OneHotEncoder for categorical columns and StandardScaler for numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols)
        ])

    return preprocessor, X_train, y_train, X_test, y_test


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
    train_df, validate_df, test_df = load_df(params, location='Sonoma')
    
