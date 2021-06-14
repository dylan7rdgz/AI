# linear regression is just a machine learning model


import numpy as np


# set of equations

random_values = np.random.rand(100, 1)
print(random_values)

# random values is our input to our dataset and X is the output, the form of the foll: equation is y = mx
X = 2 * random_values
print(X)


# what is numpy.random? it will generate random values of 100 rows and 1 column (row,col)
# how numpy.random  works? it uses a Bit generator in conjunction with a Sequence generator
# Bit Generator will manufacture a SEQUENCE of signed integer (32-64) bits and input it into the Generator
# the Generator will transform this sequence of random bits into a sequence of numbers that will follow a specific
# probability distribution(Uniform normal or binomial)
# read more: https://numpy.org/doc/stable/reference/random/index.html?highlight=rand#module-numpy.random


# difference between randn and rand?
# randn: Return a random matrix with data from the “standard normal” distribution.
# rand: Random values in a given shape


random_values_from_normal_distribution = np.random.randn(100, 1)
y = 4 + 3*X + random_values_from_normal_distribution  # y is the "target" values

X_b = np.c_[
    np.ones((100, 1)),  # generate a tensor of shape 100*1
    X  # add x0 = 1 to each instance
]  # add ones to all instances, this is what the .c_ operator does
# X_b is generated for the matrix multiplication, check line 62

print("X_b:", X_b)
# we are using .T and the compiler understands this because this is a numpy data structure
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

# the function we used to generate the data: y = 3x + 4 + Gaussian noise
# https://towardsdatascience.com/linear-regression-91eeae7d6a2e
# imagine that the Gaussian is some small addition to the actual y value
# this maybe due to rounding up, not sure

# theta_best -> obtained: [4, 3] expected: [4.215, 2.770] (due to gaussian noise)

# now that we have predicted a model that is y = 4.215x + 2.770 we can predict for a data input x what is the output y

# lets take a point from the dataset itself

print("input_data", X[0])
print("target_label", y[0])
predicted_value = 4.215*X[0] + 2.770
print("predicted data", predicted_value)
print("Error for this data point prediction", predicted_value-y[0])
print("this is the equations approach\n")

# or

X_new = np.array([[0], [2]])
X_new_b = np.c_[
    np.ones((2, 1)),
    X_new
]
# X_new_b is now [0,1] and [2,1]
# consider input instance [2,1]
# the multiplication below is [2,1]*[3
#                                    4]
#


y_predict = X_new_b.dot(theta_best)
print("this is the matrix approach")

print("predicted data", y_predict)

# this was the error_rate for one data point, lets see for



# lets see the graph of this line
# import matplot as plt
import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])  # x ranges from 0,2 and y ranges from 0 to 15
plt.show()





# performing linear regression using scikit learn

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
lin_reg.predict(X_new)



# why is pseudo inverse superior to Normal Equations?
# 1.even if the matrix XT.X is singular the pseudo inverse is still defined, handles edge cases nicely
# 2. more efficient

# what is pseudo inverse in the first place?
'''
technique used: SVD
decomposes the training matrix into multiplication of 3 sub-matrices: ie. U.P+.VT
P+ is calculated by:
    1. sets a threshold
    2. all values below this threshold are converted to 0's
    3. take inverses (^-1) of remaining values
    4. transpose this final matrix
'''

np.linalg.pinv(X_b).dot(y)


# computational complexity

# time complexity

# using normal equations
'''
Normal Equation computes the inverse of (n+1)*(n+1) matrix where n is the number of features
Computational complexity of inverting such a matrix is O(n^3) depending on the implementation
'''

# using SVD
'''
O(n^2)
'''


