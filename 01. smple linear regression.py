import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([10, 20, 30, 40, 50, 60]).reshape(-1,1)
Y = np.array([2, 4, 3, 6, 7, 9])

model=LinearRegression()

model.fit(X,Y)

Y_predict = model.predict(X)

plt.scatter(X,Y,color='blue',label='Actual Values')
plt.plot(X,Y_predict,color='red',label='Predicted values')
plt.title('Linear Regression')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

slope = model.coef_[0]
intercept = model.intercept_
print("slope= ",slope)
print("intercept= ",intercept)