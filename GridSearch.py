#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
y_test = le_Stay.fit_transform(y_test)

#encoding of independent features
objectColumns = []
for col in X:
    if X[col].dtype.name == 'object':
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.codes
        X_test[col] = X_test[col].astype('category')
        X_test[col] = X_test[col].cat.codes
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


X_test[["Admission_Deposit"]] = sc.fit_transform(X_test[["Admission_Deposit"]])
X_test[["Visitors with Patient"]] = sc.fit_transform(X_test[["Visitors with Patient"]])
X_test[["City_Code_Patient"]] = sc.fit_transform(X_test[["City_Code_Patient"]])
X_test[["Bed Grade"]] = sc.fit_transform(X_test[["Bed Grade"]])
X_test[["Available Extra Rooms in Hospital"]] = sc.fit_transform(X_test[["Available Extra Rooms in Hospital"]])
X_test[["City_Code_Hospital"]] = sc.fit_transform(X_test[["City_Code_Hospital"]])
X_test[["Hospital_code"]] = sc.fit_transform(X_test[["Hospital_code"]])



#Gridsearch for randomforest classifier with different number of trees
clf_rf = GridSearchCV(RandomForestClassifier(),{
    'n_estimators':[10,20,30,40,50,60,70,80,90]
},cv=4,return_train_score=False)
clf_rf.fit(X,y)
df = pd.DataFrame(clf_rf.cv_results_)
print(df[['param_n_estimators','mean_test_score']])

#Gridsearch for decision tree classifier with different depth values
clf_dt = GridSearchCV(DecisionTreeClassifier(),{
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12]
},cv=4,return_train_score=False)
clf_dt.fit(X,y)
df = pd.DataFrame(clf_dt.cv_results_)
print(df[['param_max_depth','mean_test_score']])


##Gridsearch for knn classifier with different number of k values
clf_knn = GridSearchCV(KNeighborsClassifier(),{
    'n_neighbors':[2,5,10,15,20,30]
},cv=4,return_train_score=False)
clf_knn.fit(X,y)
df = pd.DataFrame(clf_knn.cv_results_)
print(df[['param_n_neighbors','mean_test_score']])




