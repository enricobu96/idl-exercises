from threading import local
import numpy as np
from sklearn.metrics import confusion_matrix

"""
calculate_precision(predictions: tensor, target: tensor) -> precision: int
Input:
    - predictions: predicted labels
    - target: gold data
Output:
    - precision: average precision over the classes
"""
def calculate_precision(predictions, target):
    tp = np.intersect1d(predictions.numpy(), target.numpy())
    fp = np.setdiff1d(target.numpy(), np.setdiff1d(tp, np.union1d(predictions.numpy(), target.numpy())))
    prec = len(tp) / (len(tp) + len(fp)) if (len(tp)+len(fp)) > 0 else 0
    # print(len(tp), len(fp), prec)
    return prec

"""
calculate_recall(predictions: tensor, target: tensor) -> recall: int
Input:
    - predictions: predicted labels
    - target: gold data
Output:
    - recall: average recall over the classes
"""
def calculate_recall(predictions, target):
    tp = np.intersect1d(predictions.numpy(), target.numpy())
    fn = np.setdiff1d(predictions.numpy(), np.setdiff1d(tp, np.union1d(predictions.numpy(), target.numpy())))
    rec = len(tp) / (len(tp) + len(fn)) if (len(tp)+len(fn)) > 0 else 0
    return rec