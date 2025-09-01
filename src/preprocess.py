import pandas as pd
import os
from src.config import FOLDER

def _get_data():
    '''This function literally does what is say, it gets the data from the pre configured folder'''
    for file in os.listdir(FOLDER):
        file_path = os.path.join(FOLDER, file)
        if(file.endswith('xlsx')):
            df = pd.read_excel(file_path)
            print(f"{file_path} found as an excel file, excrating the data and proceeding to the next steps...")
        elif(file.endswith('csv')):
            df = pd.read_csv(file_path)
            print(f"{file_path} found as an csv file, excrating the data and proceeding to the next steps..")
    if df is None:
        print(f"Data file not found in the {FOLDER} path")
        return None
    return df

def _clean_data(df):
    '''At this point this function just drops the NaN values if they arent too much in the dataset,
       later I would like to automate the process of removing outliers'''
    na_values_count = df.isnull().any(axis=1).sum()
    if(df.isnull().any(axis=1).sum() < len(df)*0.2):
        #if the NaN aren't greater than 20% of the data, we kick them of
        print(f"{na_values_count} rows with missing values found in your dataset\n We'll remove them because it won't prejudice our analysis")
        df_cleaned = df.dropna()
        return df_cleaned
    print(f"{na_values_count} rows with missing values found in your dataset\n We'll NOT remove them because it would prejudice our analysis")
    return df

def preprocess_data():
    '''Just run the other two functions so we can call just one single function in the pipe file'''
    df = _get_data()
    df = _clean_data(df)
    return df