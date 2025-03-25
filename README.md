# News Classification Project

## Overview
This project focuses on classifying news articles into different categories using machine learning techniques. The dataset consists of news articles with their corresponding categories, and the model is trained to predict the category of a given news article based on its text.

## Features
- Preprocessing of text data (tokenization, stopword removal, TF-IDF transformation, etc.)
- Implementation of multiple machine learning models (e.g., Logistic Regression, Multinomial Naive Bayes, Random Forest, Gradient Boosting, Support Vector Machine, Stochastic Gradient Descent, XGBoost)
- Evaluation of model performance using accuracy, precision, recall, and F1-score
- Hyperparameter tuning to improve model performance
- Deployment script (if applicable)

## Dataset
- The dataset consists of news articles labeled with their respective categories.
- Each entry includes:
  - `Title`: The headline of the news article
  - `Topic`: The label assigned to the article


## Installation 
To set up the project, install the required dependencies using:
pip install -r requirements.txt


## Results
- Model performance (Accuracy):
  - Logistic Regression: 71.25%
  - Multinomial Naive Bayes: 71.95% 
  - XGBoost: 65.65% 
  - Random Forest: 65.45%
  - Gradient Boosting: 61.65%
  - Support Vector Machine: 71.7%
  - Stochastic Gradient Decent: 72.55%
  
## Deployment & Monitoring

CI/CD: Implemented using GitHub Actions for automated testing, building, and deployment.
Deployment: The model is deployed as a web service using [deployment method (e.g., Docker, Kubernetes, FastAPI)].
Monitoring: Prometheus collects system and application metrics, while Grafana is used for visualization and alerting.


## Future Improvements
- Improve data preprocessing and feature extraction
- Experiment with more advanced deep learning models like LSTM, BERT etc.


## Acknowledgments
- Dataset provided by DeepLearning.AI
