import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')

# print(train_features.head())
# print(train_labels.head())

#Function to print best results
def print_res(cv_output):
    print(cv_output.best_params_)
    means = cv_output.cv_results_['mean_test_score']
    stds = cv_output.cv_results_['std_test_score']
    params = cv_output.cv_results_['params']
    for means,std,params in zip(means, stds, params):
        print('mean score : {} std : +/- {} for {}'.format(round(means,3), round(std,3), params))


lr = LogisticRegression(max_iter=10000)
parameters = {
    'C' : [.01, .1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
cv = GridSearchCV(lr, parameters, cv = 5)
cv.fit(train_features, train_labels.values.ravel())
print_res(cv)

# storing best model
joblib.dump(cv.best_estimator_, 'Models/LR_Model.pkl')