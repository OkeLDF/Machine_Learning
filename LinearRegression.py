import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    '''
        #### Attributes:
        
        - `coefs`: NumPy array containing the coefficients calculated by the model

        - `metrics`: Python dictionary containing model evaluation metrics
            - `metrics['mse']`: gets the mean squared error
            - `metrics['rmse']`: gets the root mean squared error
            - `metrics['mae']`: gets the mean absolute error
            - `metrics['r2']`: gets the r2 score
    '''

    coefs = np.array([])
    metrics = {}

    def _add_bias(self, X_np):
        return np.c_[np.ones(X_np.shape), X_np]
    
    def _calculate_metrics(self, y_np, y_pred):
        self.metrics['mse'] = np.square(y_np - y_pred).mean()
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = (np.abs(y_np - y_pred)).mean()
        self.metrics['r2'] = 1 - np.divide(
            np.square(y_np - y_pred).mean(),
            np.square(y_np - y_np.mean()).mean()
        )

    def fit(self, X, y) -> None:
        X_np = np.array(X)
        y_np = np.array(y)
        X_bias = self._add_bias(X_np)

        self.coefs = np.linalg.pinv(X_bias).dot(y_np) 

        y_pred = X_bias.dot(self.coefs)
        self._calculate_metrics(y_np, y_pred)

    def predict(self, X) -> np.ndarray:
        X_np = np.array(X)
        X_bias = self._add_bias(X_np)
        y_pred = X_bias.dot(self.coefs)
        return y_pred

if __name__ == '__main__':
    def define_data(n=100, m=1, coefs=[4,3], X_range=20, outlier_factor=10):
        X = X_range * np.random.rand(n, m)
        y = coefs[0] + coefs[1] * X + outlier_factor * np.random.rand(n, m)
        return X, y
    
    def run_linreg(X, y):
        linreg = LinearRegression()
        linreg.fit(X, y)

        print('coefs   :\n  ', linreg.coefs[0], ',', linreg.coefs[1], '\n')
        print('metrics :\n  ', linreg.metrics)

        return linreg

    def run_test_plot(X, y, linreg):
        X_new = np.array([
            [np.min(X)],
            [np.max(X)]
        ])

        y_pred = linreg.predict(X_new)

        if X.shape[1] == 1:
            plt.scatter(X, y)
            plt.plot(X_new, y_pred, 'r-')
            plt.show()

    X, y = define_data()
    linreg = run_linreg(X, y)
    run_test_plot(X, y, linreg)