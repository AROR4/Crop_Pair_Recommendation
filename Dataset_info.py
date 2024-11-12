
import pandas as pd
import numpy as np
df = pd.read_csv('/Crop_recommendation.csv')
df.head()

df.isnull().sum()

df['label'].value_counts

x=df.drop('label',axis=1)
y=df['label']

x.info()

y.info()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

x_train.info()

x_test.info()

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred1=model.predict(x_test)

from sklearn.metrics import accuracy_score
logistic_reg_acc=accuracy_score(y_test,y_pred1)
print("logistic_reg_acc",str(logistic_reg_acc))

from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
y_pred3=model2.predict(x_test)

decision_acc=accuracy_score(y_test,y_pred3)
print("dicision tree acc",str(decision_acc))

from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier()
model3.fit(x_train,y_train)
y_pred4=model3.predict(x_test)

random_acc=accuracy_score(y_test,y_pred4)
print("random forest acc",str(random_acc))

