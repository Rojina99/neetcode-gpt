import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # epsilon = 1e-7
        # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cross_entropy_loss = 0

        for y_t, y_p in zip(y_true, y_pred):
            y_p = y_p + 1e-7
            cross_entropy_loss+= y_t * np.log(y_p) + (1-y_t) * np.log(1-y_p)
        cross_entropy_loss = -cross_entropy_loss
        return np.round(cross_entropy_loss/y_true.size, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        cross_entropy_loss = 0

        for y_t, y_p in zip(y_true, y_pred):
            y_p = y_p + 1e-7
            for y_t_i, y_p_i in zip(y_t, y_p):
                cross_entropy_loss+= y_t_i * np.log(y_p_i)
        cross_entropy_loss = -cross_entropy_loss

        return np.round(cross_entropy_loss/len(y_true), 4)
