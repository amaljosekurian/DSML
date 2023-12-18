from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

x=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([11,12,13,14,15,16,17,18,1])

model=LinearRegression()
model.fit(x,y)
slope=model.coef_[0]
intercept=model.intercept_
print("slope",slope)
print("intercept",intercept)
mean=mean_squared_error(y,x)
print(mean)

