import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('val_features.csv')
val_labels = pd.read_csv('val_labels.csv')

test_features = pd.read_csv('test_features.csv')
test_labels = pd.read_csv('test_labels.csv')

# Load models from .pkl files
models = {}
for mdl in ['GB', 'LR', 'MLP', 'RF', 'SVM']:
    models[mdl] = joblib.load('Models/{}_Model.pkl'.format(mdl))
# print(models)


def evaluate_model(model, model_name, features, labels):
    start = time()
    prediction = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, prediction), 3)
    precision = round(precision_score(labels, prediction), 3)
    recall = round(recall_score(labels, prediction), 3)
    time_taken = round((end - start)*1000, 2)
    print('{}--Accuracy {}--Precision {}--Recall {}--Time {}'.format(model_name,
          accuracy, precision, recall, time_taken))

# evaluating all models
# for mdl_name,mdl in models.items():
#     evaluate_model(mdl, mdl_name, val_features, val_labels)


# testing best model
evaluate_model(models['GB'], 'GB', test_features, test_labels)
