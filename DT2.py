from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt

br=load_breast_cancer
x=br.data
y=br.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(max_depth=3)
model.fit(x_train,y_train)
v=model.predict(x_test)
print(v)
a=accuracy_score(y_test,v)
print("accuracy_score",a)
b=classification_report(y_test,v)
print("classification_report",b)

plt.figure(figsize=(15,20))
plot_tree(model,filled=True,feature_names=br.feature_names,class_names=br.target_names)
plt.show()