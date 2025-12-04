import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1) small dataset (x must be 2D for sklearn)
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

# 2) fit model
model = LinearRegression().fit(x, y)

# 3) predictions for training points (to draw the fitted line)
y_pred = model.predict(x)

# 4) predict for new test inputs
x_test = np.array([[2.5], [6]])            # new points you want answers for
y_test = model.predict(x_test)
print("Predictions:", list(zip(x_test.flatten(), y_test.flatten())))

# 5) plot: scatter (data) + fitted line + test points
plt.scatter(x, y, label='data')
plt.plot(x, y_pred, label='fitted line')
plt.scatter(x_test, y_test, marker='x', s=80, label='test points')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
