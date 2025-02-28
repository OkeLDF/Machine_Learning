import numpy as np
from scipy.stats import mode

class KNNClassifier:
    _X_train = None
    _y_train = None
    k = None
    _is_fitted = None

    def __init__(self, k=7):
        if k % 2 == 0: k += 1
        self.k = k
        self._is_fitted = False

    def fit(self, X, y) -> None:
        self._X_train = X
        self._y_train = y
        self._is_fitted = True

    def _distance(self, x, x_train): # Euclidian distance
        return np.linalg.norm(x - x_train)

    def _sort_distances(self, d):
        return d[d[:,0].argsort()]

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise Exception('KNNClassifier must be fitted before making predictions')
        
        X_np = np.array(X)
        y_pred = np.array([])

        for x in X_np:
            distances = np.array([])

            for x_train in self._X_train:
                dist = self._distance(x, x_train)
                distances = np.append(distances, dist)

            distances_classes = np.c_[distances, self._y_train]
            sorted_distances = self._sort_distances(distances_classes)

            nearest = sorted_distances[:self.k,1]
            clas = mode(nearest).mode
            y_pred = np.append(y_pred, clas)

        return y_pred