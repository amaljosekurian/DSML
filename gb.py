from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

digits=load_breast_cancer()
x=digits.data
y=digits.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
v=gnb.predict(x_test)
print(v)
accuracy=accuracy_score(y_test,v)
c=classification_report(y_test,v)
print("accuracy:",accuracy)
print("classification-report",c)
