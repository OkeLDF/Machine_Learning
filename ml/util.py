import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, console_print=0):
    '''
        - `console_print`:
            - `0` to plot the confusion matrix with Matplotlib
            - `1` to print the confusion matrix in the console
            - `2` to plot and print the confusion matrix
    '''
    
    if not y_true.shape == y_pred.shape:
        raise Exception(f'Shape of y_true must be equal to shape of y_pred (y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape})')
    
    n = np.unique(y_true).size
    cm = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            cm[i,j] = np.sum((y_true == i) & (y_pred.astype(np.int8) == j))
    
    if console_print in [1, 2]:
        print('\nConfusion Matrix:')
        print(cm)
        print('Row: true classes')
        print('Col: predicted classes')

    if console_print in [0, 2]:
        plt.matshow(cm, cmap='Wistia')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted classes')
        plt.ylabel('True classes')
        for (i,j), z in np.ndenumerate(cm):
            plt.text(j, i, z, ha='center', va='center') 
        plt.show()