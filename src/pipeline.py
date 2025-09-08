import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.train import tune_and_train_model
from src.preprocess import preprocess_data
from src.config import TARGET_COLUMN, MODEL_SELECTED

def run_pipeline():
    df, numerical_columns, categorical_columns = preprocess_data(TARGET_COLUMN)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, #20/80 for test and train
                                                        random_state=42)
                                                        #stratify=y) #stratfy to guarantee fair evaluation

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

    #setting up the experiment using mlflow
    mlflow.set_experiment("ML Pipeline Experiment")
    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_SELECTED)

        #searching for models using gridsearch
        search_results = tune_and_train_model(
            X_train, y_train, preprocessor, MODEL_SELECTED
        )

        #selecting the best to log
        best_model = search_results.best_estimator_
        best_params = search_results.best_params_

        mlflow.log_params(best_params)

        #evaluating the model using the test samples
        predictions = best_model.predict(X_test)
        mlflow.sklearn.log_model(best_model, "model")
        if (MODEL_SELECTED == 'xgboost' or MODEL_SELECTED.contains('regression')):
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            print(f"Run finished.\nMSE: {mse:.4f}")
            print(f"Run finished.\nR2: {r2:.4f}")
        else: 
            accuracy = accuracy_score(y_test, predictions)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Run finished.\nAccuracy: {accuracy:.4f}")
        print("Check the MLflow UI for more details.")

if __name__ == "__main__":
    run_pipeline()
