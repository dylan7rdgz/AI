# equation of a line: y = mx + c
# theta_values are m and c
# start with some random m,c values
# y >= 0 if the label is +ve
# y < 0 if the label is -ve


# consider this data
# 16.2, 15.3 => 1
# if 16.2m + 15.3 >= 0 then this label is correctly classified
# if 16.2m + 15.3 < 0 then this label is in-correctly classified

# then what's so great about the sigmoid

# ok, so when the input is way further than 0, the probability is 1
# if the input is negatively, way further than 0, the probability is 0
# hence the sigmoid is used to convert distance to a probability

# cost function: (penalty when the input is wrongly classified) (we want to minimize the cost function)
# [label - sigma(equation output)], correct this

# similar to linear_regression, does sklearns internal implementation uses bi-diagonalization?

from sklearn import datasets
iris = datasets.load_iris()
print(list(iris.keys()))
X = iris["data"][:, 3:]

import numpy as np
y = (iris["target"] == 2).astype(np.int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)


print(log_reg.predict([[1.7], [1.5]]))

import matplotlib.pyplot as plt
plt.plot(X_new, y_proba[:, 1], "g--", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")


plt.legend()
plt.show()



