import pandas as pd

from src.preprocess import preprocess_data
from src.train import train_model

#select the model wanted based on the EAD
MODEL_SELECTED = ''

def run_pipeline():
    df = preprocess_data()

    ###In building section###

    model = train_model(x, y, x2, y2, MODEL_SELECTED)

if __name__ == "__main__":
    run_pipeline()