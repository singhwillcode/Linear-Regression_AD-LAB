import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegression().fit(x, y)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


x_test = np.array([[2.5], [6]])
y_test = model.predict(x_test)
print("Predictions for x_test:", y_test)

plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.show()
