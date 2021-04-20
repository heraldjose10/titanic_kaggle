import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

titanic = pd.read_csv('titanic.csv')

# replace sex with number-----------------------
sex = {'male': 0, 'female': 1}
titanic.Sex = [sex[item] for item in titanic.Sex]
# print(titanic.head())

# remove null values from Age column-----------------------
# print(titanic.isnull().sum())
titanic.Age.fillna(titanic.Age.mean(), inplace=True)
# print(titanic.head())

# adding parents and siblings to fam column--------------------
titanic['Family'] = titanic.SibSp + titanic.Parch

#drop reduntant columns------------------------------------------
titanic.drop(['SibSp', 'Parch', 'PassengerId'], axis=1 ,inplace=True)
# print(titanic.head())

#fixing cabin null values----------------------------------
# print(titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean())
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
# print(titanic.head())

# drop cols----------------------------------------------------------------
titanic.drop(['Name', 'Cabin', 'Ticket', 'Embarked'], axis=1 ,inplace=True)
# print(titanic.head())

# save to csv file-------------------------------
titanic.to_csv('titanic_clean.csv', index=False)

