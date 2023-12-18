from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=45)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
t=knn.predict(x_test)
print(t)
accuracy=accuracy_score(y_test,t)
print(accuracy)