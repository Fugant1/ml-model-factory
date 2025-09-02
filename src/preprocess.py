import pandas as pd
import numpy as np
import os
from src.config import FOLDER, FILENAME

def _get_data() -> pd.DataFrame:
    '''This function literally does what is say, it gets the data from the pre configured folder'''
    file_path = os.path.join(FOLDER, FILENAME)

    if not os.path.exists(file_path):
        print(f"Error: File '{FILENAME}' not found in folder '{FOLDER}'.")
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    df = None
    if file_path.endswith('.xlsx'):
        print(f"Loading Excel file: {file_path}")
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type for: {FILENAME}. Please use .csv or .xlsx.")

    print("Data loaded successfully.")
    return df

def _clean_data(df : pd.DataFrame) -> pd.DataFrame:
    '''At this point this function just drops the NaN values if they arent too much in the dataset,
       later I would like to automate the process of removing outliers'''
    rows_with_na = df.isnull().any(axis=1).sum()

    if(0 < rows_with_na < len(df)*0.2):
        #if the NaN aren't greater than 20% of the data, we kick them of
        print(f"{rows_with_na} rows with missing values found in your dataset\n We'll remove them because it won't prejudice our analysis")
        df = df.dropna()
    elif rows_with_na == 0:
        print("No missing values found in the dataset.")
    else:
        print(f"{rows_with_na} rows with missing values found in your dataset\n We'll NOT remove them because it would prejudice our analysis")
    
    print("Data cleaned successfully")
    return df

def _organize_data(df : pd.DataFrame, nunique_threshold=6 ) -> pd.DataFrame:
    '''This function just organize and divide data into categorical and numerical columns to help analyzing the data and training the model after this'''
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    newly_categorical = []

    for col in numerical_cols:
        if df[col].nunique() <= nunique_threshold: #here we have a limit to classify false numerical data
            newly_categorical.append(col)
            
    numerical_cols = [col for col in numerical_cols if col not in newly_categorical]
    categorical_cols.extend(newly_categorical)

    print("Data organized successfully")
    return df, numerical_cols, categorical_cols

def preprocess_data():
    '''Just run the other two functions so we can call just one single function in the pipe file'''
    df = _get_data()
    df = _clean_data(df)
    df, numerical_cols, categorical_cols = _organize_data(df)

    print("Preprocess of data done!")
    return df, numerical_cols, categorical_cols