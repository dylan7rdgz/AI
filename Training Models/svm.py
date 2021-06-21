# versatile Machine Learning Model

# used for small- and medium-sized data sets, why?

# what is large margin classification?
'''
  our main intention is to keep the line(decision-boundary) as far away from the closest training instances
  We can think of an SVM classifier as fitting the widest possible street between the classes
'''

# scaled vs un-scaled features(use sk-learns StandardScaler), how it affects svm's

# hard margin classification
# hard margin classification requires that the instances should not be inside the street or on the wrong side
# as mentioned our main intention is to make use of the support vectors
# if we enforce hard margin classification, our street will end up becoming very small and hence will not able
# to generalize well on new training instances

# soft margin classification is more flexible, it allows few instance to be in the street and also on the wrong side
# so tht it helps the learning algorithm to find the widest possible gutter


# import numpy as np
# from sklearn import datasets
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC # Support Vector Classifier
#
# iris = datasets.load_iris()
# print(iris["data"])
# X = iris["data"][:, (2, 3)]  # petal length, petal width
# print(X)
# y = (iris["target"] == 2).astype(np.float64) # Iris virginica
#
# svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("linear_svc", LinearSVC(C=1, loss="hinge"))
# ])
#
# svm_clf.fit(X, y)
#
# prediction = svm_clf.predict([[5.5, 1.7]])
#
# print(prediction)


# Non linear SVM classification
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=1000, noise=0.15)

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge")),
])

#/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/sklearn/svm/_base.py:976: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
# warnings.warn("Liblinear failed to converge, increase

polynomial_svm_clf.fit(X, y)
