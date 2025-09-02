# ML Model Factory 🏭

An automated, end-to-end machine learning pipeline for training, evaluating, and versioning models. This project was built to demonstrate a foundational understanding of MLOps principles.

---

## Project Overview

This pipeline automates the entire machine learning workflow. It starts by ingesting raw data, proceeds to clean and preprocess it, trains a machine learning model, tunes its hyperparameters, and finally versions the resulting model and its performance metrics.

---

## Features

* **Automated Data Preprocessing:** Handles missing values, scales numerical features, and encodes categorical features.
* **Model Training:** Trains a classification/regression model on the prepared data.
* **Hyperparameter Tuning:** Automatically finds the best hyperparameters using techniques like Grid Search.
* **Model Versioning:** Uses MLflow to track experiments, log metrics, and version models.

---

## Tech Stack

* **Language:** Python 3.9
* **Data Handling:** Pandas, NumPy
* **ML Framework:** Scikit-learn
* **Experiment Tracking:** MLflow
* **Orchestration:** (e.g., Python scripts, GitHub Actions, etc.)

---

## How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/Fugant1/ml-model-factory.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main pipeline script:
    ```bash
    python src/pipeline.py
    ```
