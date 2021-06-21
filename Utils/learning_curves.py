from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])  # m is the current training set size which is plotted on x axis
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict[:m]))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


X = 2*np.random.rand(100, 1)

m = 100
X = 6 * np.random.rand(m, 1) - 3

# general equation of a polynomial: y = ax^2 + bx + c
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)


from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
plt.xlim(0, 80)
plt.ylim(0, 3)
plot_learning_curves(linear_regression, X, y)





from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])


# !not setting the y limits caused the val and train error to look like its close to 0
# by setting the max y limit the curves are properly plotted
plt.xlim(0, 80)
plt.ylim(0, 3)
plot_learning_curves(polynomial_regression, X, y)


