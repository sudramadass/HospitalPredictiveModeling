# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, RocCurveDisplay

def plot_confusion(y_true, y_pred, labels=(0,1), title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc(model, X_test, y_test):
    try:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.show()
    except Exception as e:
        print("Could not plot ROC:", e)
