
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 8 08:40:42 2020

@author: Arunima
""""
import pandas as pd

df = pd.read_csv('C:/Users/Arunima/OneDrive/Documents/ESCP/Python/Assg 2&3/train.csv')
df.describe()

#Checking for correlation with the dependent variable
df.corr()[1:2]

#purchaseTime, visitTime, hour since have a high correlation with our purchase and ID column is not used.

df2 = df.drop(['id', 'purchaseTime', 'visitTime','hour'], axis=1)

X = df2.loc[:, df2.columns != "label"]

y = df2.loc[:, df2.columns == "label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 0, stratify = y)

# Over sampling the data

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = 'minority', random_state = 123)
X_train, y_train = sm.fit_resample(X_train, y_train)

#Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=123, n_estimators=60)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict_proba(X_test)
y_pred_rfc = rfc.predict(X_test)

print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred_rfc))

from sklearn.model_selection import cross_val_score
from numpy import mean
scores = cross_val_score(rfc, X, y, scoring='roc_auc',n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))

#Testing the model

df_test = pd.read_csv('C:/Users/Arunima/OneDrive/Documents/ESCP/Python/Assg 2&3/test.csv')

df_test.head()

del df_test['label']

df_test = df_test.drop(['id', 'purchaseTime', 'visitTime','hour'], axis=1)

prediction = pd.DataFrame(rfc.predict_proba(df_test), columns=['Prob_0','Prob_1'])
df_test['Prob_0'] = prediction['Prob_0']
df_test['Prob_1'] = prediction['Prob_1']

Result = pd.DataFrame()

Result['Prob_0'] = df_test['Prob_0']

Result['Prob_1'] = df_test['Prob_1']

Result.to_csv("Prediction_Results.csv")

