# polynomial regression is a machine learning model


import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------generate test data--------------------------------------- #
X = 2*np.random.rand(100, 1)

m = 100
X = 6 * np.random.rand(m, 1) - 3

# general equation of a polynomial: y = ax^2 + bx + c
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
# ---------------------------------------------------------------------------------------------- #


# ----------------------------- Concept of polynomial regression -------------------------------- #
'''
observe the graph, a straight line will never fit this data
surprisingly we can use a linear-model to train non-linear data with the help of polynomial features!
add the square of every feature as a new feature, so in this example for every input in x^2 and x there is an output
so this is a kind of technique which increases the dimension to solve the existing problem

because we computed x^2 and added these value in a new axis we will get many theta values as predictor coeff's
for example theta = [13, 17]
then the equation of the line is y = 13x^2 + 17x + 5
or       equation of the line is y = 13 a  + 17b + 5
once we fit a line in this multidimensional axes, we can predict a value for (a,b)
but suppose we know a but don't know b, then we cannot predict the line
but b is a function of a which is a function of x in this case and since we know x we can predict the output
this kind of dimensionality stretch is often used in machine-learning, for example in the XOR problem, which is solved
with the help of neural network model
'''
# ---------------------------------------------------------------------------------------------- #



# ----------------------------- work in a higher dimension ------------------------------------ #

from sklearn.preprocessing import PolynomialFeatures
# be ware of the combinatorial explosion of the number of features
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features_300 = PolynomialFeatures(degree=32, include_bias=False, interaction_only=False)
#  if interaction_only is True then only degree distinct features are multiplied ans stored in the combination list

X_poly_2 = poly_features_2.fit_transform(X)  # fit to data, then transform it



X_poly_300 = poly_features_300.fit_transform(X)  # fit to data, then transform it

# ---------------------------------------------------------------------------------------------- #



# ------------------ Linear regression algorithm working on X_poly data ------------------------ #

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression_2 = linear_regression.fit(X_poly_2, y)
# check inside linear regression fit() method, they must be using some gradient descent algorithms
# ok, so I've gone through the internal working of .fit() in sklearn,
# it uses lsqr algorithm to fit the line
# so my question was does it use any gradiant descent algorithm, BUT i don't think so, it uses bidiagonalization
# algorithm, which is an iterative process, so should I even learn SGD, these internal implementations work pretty well
# what if somehow I want to override this converging algorithm using stochastic gradient descent, would I have to write
# my own implementation, because sklearns interface doesn't even allow me to send a parameter stating which converging
# algorithm i want to use
# https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf


# scikit has certain predictors that make use of stochastic gradient descent




# now suppose we use gradient descent we also have the ability to use special techniques where we can now in which
# direction the gradient moves the fastest by encorporating the second derivative tests

# but the cost function w.r.t to theta values will always be a bowl and convex and "uniform", so there is no need for a
# second derivative test


theta_2 = np.concatenate(( np.flip(linear_regression_2.coef_), linear_regression_2.intercept_), axis=None)
print("Coefficients(2):", linear_regression_2.coef_)
# print("Coefficients(2) Rev:", linear_regression_2.coef_)
print("Intercepts(2)", linear_regression_2.intercept_)
print("Theta(2)", theta_2)

# y_generated = 0.5469 * X ** 2 + 0.99 * X + 1.78 + np.random.randn(m, 1)
# y_generated_polynomial = theta_2 * X


# X_new = np.array([[-3], [2], [3]])
# X_new_p_b = np.c_[np.ones((3, 1)), X_new, X_new ** 2]
# print("X_new_p_b", X_new_p_b)
# theta_2_prime = np.c_[linear_regression_2.intercept_, linear_regression_2.coef_]
# print("Theta(2)", theta_2_prime.T) # hack of using T
# generated_outputs = theta_2_prime.dot(X_new_p_b)
# print("Y generated polynomial", generated_outputs)
#
# x_input = np.array(
#     [
#         [-3], [2], [3]
#     ]
# )
plt.xlim(-5, 5)
plt.ylim(-5, 15)
#
# plt.plot(X, y, "b.")
#
# print(X[0][0], X[1][0])
# plt.plot(
#     [X[0][0], X[1][0], X[2][0]],
#     [generated_outputs[0][0], generated_outputs[0][1], generated_outputs[0][2]])
#
# plt.show()

# generated_polynomial = np.poly1d(theta_2)
my_input = np.linspace(-100, 100, 100000)

plt.plot(X, y, "b.")

generated_polynomial = np.poly1d(theta_2)
print("Y generated polynomial", generated_polynomial)

print("my_input", my_input, "generated_polynomial:", generated_polynomial(my_input))
plt.plot(my_input, generated_polynomial(my_input), "-r")


linear_regression_300 = linear_regression.fit(X_poly_300, y)
theta_300 = np.concatenate((np.flip(linear_regression_300.coef_), linear_regression_300.intercept_), axis=None)


print("Coefficients(300):", linear_regression_300.coef_)
# print("Coefficients(300) Rev:", linear_regression_300.coef_)
print("Intercepts(300)", linear_regression_300.intercept_)

generated_polynomial = np.poly1d(theta_300)
print("Y generated polynomial", generated_polynomial)
plt.plot(my_input, generated_polynomial(my_input), "-g")

# x_input = np.array(
#     [
#        [-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]
#     ]
# )
#
# evaluated_polynomial_at_7_pts = np.array([
# [generated_polynomial(-4)],
# [generated_polynomial(-3)],
# [generated_polynomial(-2)],
# [generated_polynomial(-1)],
# [generated_polynomial(0)],
# [generated_polynomial(1)],
# [generated_polynomial(2)],
# [generated_polynomial(3)],
# [generated_polynomial(4)],
#
# ])
# plt.plot(x_input, evaluated_polynomial_at_7_pts, "g-")

plt.show()

# ---------------------------------------------------------------------------------------------- #





# ------------------------------- Performance -------------------------------------------------- #



































