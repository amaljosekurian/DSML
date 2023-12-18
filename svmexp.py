from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty=fetch_20newsgroups(shuffle=True,random_state=42,subset="train")
vector=TfidfVectorizer()
X_trainTFid=vector.fit_transform(twenty.data)
y_train=twenty.target

x_train,x_test,y_train,y_test=train_test_split(X_trainTFid,y_train,test_size=0.3,random_state=42)
svm=SVC(kernel="linear",random_state=42)
v=svm.fit(x_train,y_train)
a=svm.predict(x_test)
accuracy=accuracy_score(y_test,a)
print("accuracy_score",accuracy)
classification=classification_report(y_test,a)
print("classification_report",classification)

new_data=["I have a question about computer graphics","This is a medical related topic"]
c=vector.transform(new_data)
b=svm.predict(c)
for i in range(len(new_data)):
    predict=twenty.target_names[b[i]]
    print(predict)