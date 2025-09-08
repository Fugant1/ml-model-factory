from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
import os

#Folder of data acquisition
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FOLDER = os.path.join(PROJECT_ROOT, 'data')

#Filename to be analyzed
FILENAME = 'dataset.csv'

#Select the model wanted based on the EDA
MODEL_SELECTED = 'xgboost'

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
    },
    'xgboost': {
        'model': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, tree_method="hist")
    }
    #Later I'll add here XGBoost and stuff, let's try it simple for now
}
