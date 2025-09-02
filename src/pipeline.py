import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.preprocess import preprocess_data
from src.train import train_model

def run_pipeline():
    df, numerical_columns, categorical_columns = preprocess_data()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, #20/80 for test and train
                                                        random_state=42, 
                                                        stratify=y) #stratfy to guarantee fair evaluation

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            #standard scaler to avoid scale problems while the training section
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            #one hot encoder to guarantee that the model will not mess up with categorical data
            #handle_unknown='ignore' this flag is to the training section, since we are using cross validation
            #we need to have a way to not crash the training if some feature was never seen during the train and validation
        ]
    )

    trained_model = train_model(X_train, X_test, y_train, y_test, preprocessor, MODEL_SELECTED)

if __name__ == "__main__":
    run_pipeline()
