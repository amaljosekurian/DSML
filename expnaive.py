from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
gb=GaussianNB()
a=gb.fit(x_train,y_train)
v=gb.predict(x_test)
accuracy=accuracy_score(y_test,v)
print("accuracy:",accuracy)
c=classification_report(y_test,v)
print("classification report:",c)