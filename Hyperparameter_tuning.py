# -*- coding: utf-8 -*-
"""Hyperparameter_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lvTTHUgZKXr1jkM661V9ExYo9NI6lfrr
"""

#importing libraries
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


#reading file using pandas
train_data = pd.read_csv("train_data.csv")

#deleting rows having missing values
train_data.dropna(how='any', inplace=True)

#separating independent features
X = train_data.iloc[:, 1:-1]

#deleting irrelavent feature
X = X.drop(["patientid"], axis = 1)

#separating target column
y = train_data.iloc[:, -1]

#simple train/test split
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.33, random_state = 7)

#encoding of target features
le_Stay = LabelEncoder()
y = le_Stay.fit_transform(y)



#encoding of independent features
objectColumns = []
for col in X:
    if X[col].dtype.name == 'object':
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.codes
        objectColumns.append(col)


#Normalization
sc = MinMaxScaler()
X[["Admission_Deposit"]] = sc.fit_transform(X[["Admission_Deposit"]])
X[["Visitors with Patient"]] = sc.fit_transform(X[["Visitors with Patient"]])
X[["City_Code_Patient"]] = sc.fit_transform(X[["City_Code_Patient"]])
X[["Bed Grade"]] = sc.fit_transform(X[["Bed Grade"]])
X[["Available Extra Rooms in Hospital"]] = sc.fit_transform(X[["Available Extra Rooms in Hospital"]])
X[["City_Code_Hospital"]] = sc.fit_transform(X[["City_Code_Hospital"]])
X[["Hospital_code"]] = sc.fit_transform(X[["Hospital_code"]])




#making dictionary of models with different parameters to pass in the RandomizedSearchCV to find best model for our dataset
model_params = {
    
    'DecisionTree':{
        'model':DecisionTreeClassifier(criterion='gini'),'params':{'max_depth':[7,9,11] }
    },
    'RandomForest':{
        'model':RandomForestClassifier(),'params':{'n_estimators':[20,50,70]}
    },
    'KNN':{
        'model': KNeighborsClassifier(),'params':{'n_neighbors':[10,20,30]}
    }

}

#logic for finding best model 
scores = []
for model_name,mp in model_params.items():
    clf = RandomizedSearchCV(mp['model'],mp['params'],cv=2,return_train_score=False,n_iter=2)
    clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_param':clf.best_params_
    })

#Dataframe having models with best accuracy with their best parameter value
final_df = pd.DataFrame(scores,columns=['model','best_score','best_param'])
print(final_df)

import matplotlib.pyplot as plt
import seaborn as sns

#Refeerence:: https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
#For displaying the values on the graph
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

#It take's too much time so i have taken value from the results and made the graph from it
#For Decision Tree: different depth of tree hyperparameter tuning
accuracy = [30.42,35.59,36.81,37.68,39.22,39.60,40.08,40.60,40.76,40.90,40.68,40.26]

plt.rcParams.update({'font.size': 20})

x_label = list(range(1,13))

plt.figure(figsize = (15, 10))
ax=sns.barplot(x=x_label, y=accuracy)
show_values_on_bars(ax)
plt.title('Decision Tree With Different Depth')
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy')

#For Random Forest: different number of trees hyperparameter tuning
accuracy = [35.89,37.00,37.52,37.74,37.79,38.10,38.14,38.22,38.29]
x_label = list(range(10,100,10))


plt.figure(figsize = (10, 5))
ax=sns.barplot(x=x_label, y=accuracy)
show_values_on_bars(ax)
plt.title('Random Forest With Different number of trees')
plt.xlabel('number of Trees')
plt.ylabel('Accuracy')

#For KNN: different value of 'k' hyperparameter tuning
accuracy = [27.45,28.79,30.21,30.59,30.93,31.38]
x_label = [2,5,10,15,20,30]

plt.figure(figsize = (10, 5))
ax=sns.barplot(x=x_label, y=accuracy)
show_values_on_bars(ax)
plt.title('KNN With Different value of k')
plt.xlabel('value of k')
plt.ylabel('Accuracy')