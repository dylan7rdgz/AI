# bagging uses the same learning algorithm but makes predictions in different batches of training instances
# choose the majority votes of each ensemble

# when replacement = False, bagging = pasting

# think of picking up training instances from a bag
# there can be a case where we put back the data in the bag and pick up another bunch of data which may include
# a subset of the old data

# bagging may contribute to higher regularization, not sure!!

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=100000, noise=0.15)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,  # 500 DT's
    max_samples=100,  # train them by picking 100 training instances at a time
    bootstrap=True,   # bootstrap=False => pasting
    n_jobs=-1
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
