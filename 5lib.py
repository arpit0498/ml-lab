import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data=pd.read_csv("mush.csv")
df=pd.DataFrame(data)
df
df.replace('?',np.nan,inplace=True)
df.dropna(axis=1,inplace=True)
df
df=pd.get_dummies(df)
df
Y=df.iloc[:,0]
X=df.iloc[:,2:]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
num_training=y_train.shape[0]
num_test=y_test.shape[0]
print('training:'+str(num_training)+',test:'+str(num_test))
from sklearn.naive_bayes import GaussianNB

gnb =GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
print("Misclassified samples:%d out of %d" %((y_test!=y_pred).sum(),y_test.shape[0]))

print('Accuracy:'+str(accuracy_score(y_test,y_pred)))