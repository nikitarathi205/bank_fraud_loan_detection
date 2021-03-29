import numpy as np
import pandas as pd                                                    # data processing, CSV file I/O (e.g. pd.read_csv)
df=pd.read_csv('train_indessa.csv')
x=pd.DataFrame(df)
x = df.iloc[:, 0:44]
y = df.iloc[:, 44]
d={'36 months':36, '60 months':60}
x.term=x.term.map(d)
d1={'OWN':0, 'MORTGAGE':1, 'RENT':2}
x.home_ownership=x.home_ownership.map(d1)
d2={'Non Verified':0, 'Source Verified':1, 'Verified':2}
x.verification_status=x.verification_status.map(d2)
d3={'f':0,'w':1}
x.initial_list_status=x.initial_list_status.map(d3)
x=x.replace(np.nan,0)
x.drop(['batch_enrolled','grade','sub_grade','emp_title','emp_length','pymnt_plan','desc','purpose','title','zip_code',
        'addr_state','application_type','verification_status_joint','last_week_pay'],
       axis=1,inplace=True)

print(x)

#Dividing data for tarining and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print('There are {} samples in the training set and {} samples in the test set'.format(x_train.shape[0], x_test.shape[0]))

#svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# random forest model creation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train_std, y_train)
# predictions
y_pred = rfc.predict(x_test_std)
y_train_pred=rfc.predict(x_train_std)

print('The accuracy of the RF classifier on training data is {:.2f}'.format(accuracy_score(y_train,y_train_pred)))
print('The accuracy of the RF classifier on test data is {:.2f}'.format(accuracy_score(y_test,y_pred)))

#roc computation
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
roc_auc=auc(fpr,tpr)
print('The auc roc of the RF classifier on test data is',roc_auc)

#Regressor model creation
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train_std, y_train)
y_train_pred=reg.predict(x_train_std)
y_pred = reg.predict(x_test_std)
y_pred2=[]
for i in range(0,len(y_pred),1):
    if abs(y_pred[i]>=0.5):
        y_pred2.append(1)
    else:
        y_pred2.append(0)
y_pred4=[]
for i in range(0,len(y_train_pred),1):
    if abs(y_train_pred[i]>=0.5):
        y_pred4.append(1)
    else:
        y_pred4.append(0)
print('The accuracy of the LR classifier on training data is {:.2f}'.format(accuracy_score(y_train,y_pred4)))
print('The accuracy of the LR classifier on test data is {:.2f}'.format(accuracy_score(y_test,y_pred2)))

#roc computation
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
roc_auc=auc(fpr,tpr)
print('The auc roc of the LR classifier on test data is', roc_auc)
