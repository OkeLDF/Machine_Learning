from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

from ml.util import plot_confusion_matrix
from ml.knn import KNNClassifier

def define_data():
    iris = load_iris()

    # 150 instances
    X = iris['data']
    y = iris['target']

    # 120 instances
    X_train = np.r_[X[10:50], X[60:100], X[110:]]
    y_train = np.r_[y[10:50], y[60:100], y[110:]]

    # 30 instances
    X_test = np.r_[X[0:10], X[50:60], X[100:110]]
    y_test = np.r_[y[0:10], y[50:60], y[100:110]]

    return X_train, y_train, X_test, y_test

def run_knn(X, y, X_test):
    knn = KNNClassifier()
    knn.fit(X, y)
    return knn.predict(X_test)

def plot_graphs(X, y, X_test):
    fig, ax = plt.subplots(2, 2, figsize=[7,7])

    ax[0,0].scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
    ax[0,0].scatter(X_test[:,0], X_test[:,1], marker='x')

    ax[1,0].scatter(X[:,2], X[:,3], c=y, cmap='rainbow')
    ax[1,0].scatter(X_test[:,2], X_test[:,3], marker='x')

    ax[0,1].scatter(X[:,0], X[:,2], c=y, cmap='rainbow')
    ax[0,1].scatter(X_test[:,0], X_test[:,2], marker='x')

    ax[1,1].scatter(X[:,1], X[:,3], c=y, cmap='rainbow')
    ax[1,1].scatter(X_test[:,1], X_test[:,3], marker='x')

    plt.show()

def plot_classification(X, y, cmap='rainbow'):
    fig, ax = plt.subplots(2, 2, figsize=[7,7])

    ax[0,0].scatter(X[:,0], X[:,1], c=y, cmap=cmap)
    ax[1,0].scatter(X[:,2], X[:,3], c=y, cmap=cmap)
    ax[0,1].scatter(X[:,0], X[:,2], c=y, cmap=cmap)
    ax[1,1].scatter(X[:,1], X[:,3], c=y, cmap=cmap)

    plt.show()

X, y, X_test, y_test = define_data()
plot_graphs(X, y, X_test)
y_pred = run_knn(X, y, X_test)

X_total = np.r_[X, X_test]
y_pred_total = np.r_[y, y_pred]
y_test_total = np.r_[y, y_test]

# instances classified by KNN
plot_classification(X_total, y_pred_total)

# errors in classification (in red)
plot_classification(X_total, y_pred_total == y_test_total, cmap='Set1')

# plot and print consufion matrix
plot_confusion_matrix(y_test, y_pred, console_print=2)