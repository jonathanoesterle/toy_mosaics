import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode


def plot_blobs(X, y, y_true=None, assume_same_names=False):
    """Plot 2D feature blobs colored by class."""

    y_plot = y.copy()

    if not assume_same_names and y_true is not None:
        label_mapping = {}
        for label in np.unique(y):
            true_labels = y_true[y == label]
            if len(true_labels):
                label_mapping[label] = mode(true_labels, keepdims=False).mode
            else:
                label_mapping[label] = label

        y_plot = np.vectorize(label_mapping.get)(y)

    plt.figure()

    for label in np.unique(y_plot):
        mask = y_plot == label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f"Class {label}"
        )

    if y_true is not None:
        misclassified = y_plot != y_true
        plt.scatter(
            X[misclassified, 0],
            X[misclassified, 1],
            facecolors='none',
            edgecolors='k',
            label='Misclassified'
        )

    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Simulated Feature Blobs")
    plt.show()
