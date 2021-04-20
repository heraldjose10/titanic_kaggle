import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')

# Function to print best results


def print_res(cv_output):
    print(cv_output.best_params_)
    means = cv_output.cv_results_['mean_test_score']
    stds = cv_output.cv_results_['std_test_score']
    params = cv_output.cv_results_['params']
    for means, std, params in zip(means, stds, params):
        print('mean score : {} std : +/- {} for {}'.format(round(means, 3),
              round(std, 3), params))

rf = RandomForestClassifier()
parameters = {
    'max_depth' : [2, 4, 8, 16, 32, None],
    'n_estimators' : [5, 50, 250]
}
cv = GridSearchCV(rf, parameters, cv=5)
with joblib.parallel_backend('threading', n_jobs=-1):
    cv.fit(train_features, train_labels.values.ravel())
    print_res(cv)

joblib.dump(cv.best_estimator_, 'Models/RF_Model.pkl')