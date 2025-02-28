import numpy as np

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

    coefs = None
    metrics = {}
    _is_fitted = None

    def __init__(self):
        self._is_fitted = False

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
        self._is_fitted = True

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise Exception('LinearRegression must be fitted before making predictions')
        
        X_np = np.array(X)
        X_bias = self._add_bias(X_np)
        y_pred = X_bias.dot(self.coefs)
        return y_pred