import pandas as pd
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('titanic_clean.csv')
# print(titanic.head())

features = titanic.drop('Survived', axis=1)
labels = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train.to_csv('train_features.csv', index=False)
X_test.to_csv('test_features.csv', index=False)
X_val.to_csv('val_features.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)
y_test.to_csv('test_labels.csv', index=False)
y_val.to_csv('val_labels.csv', index=False)
