from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import MODELS

def _train_model(X_train, X_test, y_train, y_test, preprocessor, model_name):
    '''This function will just train the model based on the train sample provided'''
    model_config = MODELS[model_name]
    model_object = model_config['model']
    model_params_to_grid = model_config['params']

    #sets the pipeline of what to do with the preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_object)
    ])

    #divide exactly what the grid_search will do based on the params provided in the config seciton
    pipeline_param_grid = {f'model__{key}': value for key, value in model_params_to_grid.items()}

    try:
        grid_search = GridSearchCV(pipeline, 
                                param_grid=pipeline_param_grid, 
                                cv=5, scoring='accuracy',
                                n_jobs=-1) #we will use all the cpu cores to run this
    except Exception as e:
        print(f'Error while making the grid_search {e.with_traceback}')
        return
    
    grid_search.fit(X_train, y_train)

    return grid_search

def tune_and_train_model(X_train, y_train, preprocessor, model_name):
    results = _train_model(X_train, y_train, preprocessor, model_name)
    return results