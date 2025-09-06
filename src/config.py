from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Folder of data acquisition
FOLDER = 'data'

#Filename to be analyzed
FILENAME = 'train.csv'

#Select the model wanted based on the EDA
MODEL_SELECTED = 'random_forest'

#Select the target column based on the EDA
TARGET_COLUMN = ''

#Possible Models to use in each case and their parameters
MODELS = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1.0, 10],
            'solver': ['liblinear']
        }
    }
    #Later I'll add here XGBoost and stuff, let's try it simple for now
}
