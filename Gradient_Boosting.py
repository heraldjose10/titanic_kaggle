from inspect import Parameter
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

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


gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [1, 3, 4, 5, 7, 9],
    'learning_rate': [.01, .1, 1, 10, 100]
}
cv = GridSearchCV(gb, parameters, cv=5)
with joblib.parallel_backend('threading', n_jobs=-1):
    cv.fit(train_features, train_labels.values.ravel())
    print_res(cv)
joblib.dump(cv.best_estimator_, 'Models\GB_Model.pkl')
