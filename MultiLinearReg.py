import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from  sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

california_housing=fetch_california_housing()
df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)

df['target']=california_housing.target

x=df.drop("target",axis=1)
y=df['target']

L=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
v=L.fit(x_train,y_train)
p=L.predict(x_test)
print(p)
mean=mean_squared_error(y_test,p)
print(mean)
