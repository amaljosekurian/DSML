from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits=load_digits()
x=digits.data
y=digits.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=45)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
a=knn.predict(x_test)
print(a)
accuracy=accuracy_score(y_test,a)
print("accuracy",accuracy)