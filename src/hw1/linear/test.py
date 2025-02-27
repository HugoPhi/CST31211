import numpy as np


def f(X: np.ndarray, W: np.ndarray, b):
    return X @ W.T + b


def svm(score, y, delta):
    margins = score - score[np.arange(y.shape[0]), y][:, None] + delta
    margins[np.arange(y.shape[0]), y] = 0

    return np.sum(margins, axis=1)


def relu_svm(score, y, delta):
    margins = np.maximum(0, score - score[np.arange(y.shape[0]), y][:, None] + delta)
    margins[np.arange(y.shape[0]), y] = 0

    return np.sum(margins, axis=1)


def dw_svm(X: np.ndarray, y: np.ndarray):
    print(np.eye(y.shape[1])[y])
    exit(0)
    mask = ~(np.eye(y.shape[1])[y])

    dW = X[:, :, None] * mask[:, None, :]
    return dW


socre = np.array([[3.2, 5.1, -1.7],
                  [1.3, 4.9, 2.0],
                  [2.2, 2.5, -3.1]])

y = np.array([0, 1, 2])

# print(svm(socre, y, 1.0))
print(relu_svm(socre, y, 1.0))
