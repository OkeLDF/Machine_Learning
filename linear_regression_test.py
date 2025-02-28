import matplotlib.pyplot as plt
import numpy as np

from ml.linear_regression import LinearRegression

def define_data(n=100, m=1, coefs=[4,3], X_range=20, outlier_factor=10):
    X = X_range * np.random.rand(n, m)
    y = coefs[0] + coefs[1] * X + outlier_factor * np.random.rand(n, m)

    X_new = np.array([
        [np.min(X)],
        [np.max(X)]
    ])

    return X, y, X_new

def run_linreg(X, y, X_new):
    linreg = LinearRegression()
    linreg.fit(X, y)
    y_pred = linreg.predict(X_new)

    print('coefs   :\n  ', linreg.coefs[0], ',', linreg.coefs[1], '\n')
    print('metrics :\n  ', linreg.metrics)

    plt.scatter(X, y)
    plt.plot(X_new, y_pred, 'r-')
    plt.show()

X, y, X_new = define_data()
linreg = run_linreg(X, y, X_new)