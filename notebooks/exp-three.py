import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/bhauryal7/NewsClassification.mlflow"
dagshub.init(repo_owner="bhauryal7", repo_name="NewsClassification", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Logistic Regression Hyperparameter Tuning")


# ========================== TEXT PREPROCESSING ===============================
# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['title'] = df['title'].apply(lower_case)
        df['title'] = df['title'].apply(remove_stop_words)
        df['title'] = df['title'].apply(removing_numbers)
        df['title'] = df['title'].apply(removing_punctuations)
        df['title'] = df['title'].apply(removing_urls)
        df['title'] = df['title'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise 

# ==========================
# Load & Prepare Data
# ==========================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    df = normalize_text(df)
    df["topic"] = label_encoder.fit_transform(df["topic"])
    X = vectorizer.fit_transform(df["title"])
    y = df["topic"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    

    param_grid = {
    'C': [1, 10, 100],  # Inverse regularization strength
    'penalty': ['l2'],              # L2 regularization (default)
    'solver': ['sag', 'lbfgs'],     # Solvers for large datasets
    'max_iter': [2000],         # Increase if not converging
    'class_weight': [None, 'balanced']  # Handle class imbalance
}

    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred,average='weighted'),
                    "recall": recall_score(y_test, y_pred,average='weighted'),
                    "f1_score": f1_score(y_test, y_pred,average='weighted'),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_accuracy = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy )
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best Accuracy: {best_accuracy:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test) = load_and_prepare_data("notebooks/backup.csv")
    train_and_log_model(X_train, X_test, y_train, y_test)