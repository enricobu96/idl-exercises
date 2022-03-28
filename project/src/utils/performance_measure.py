from threading import local
import numpy as np

"""
precision_recall_f1(predictions: tensor, target: tensor) -> prec: float, rec: float, f1: float
Input:
    - predictions: predicted labels
    - target: gold data
Output:
    - prec: average precision over the classes
    - rec: average recall over the classes
    - f1: average f1 over the classes
"""
def precision_recall_f1(predictions, target):
    predictions = predictions.numpy().tolist()
    predictions = [tuple(x) for x in predictions]
    target = target.numpy().tolist()
    target = [tuple(x) for x in target]

    fp = set(predictions) - set(target)
    fn = set(target) - set(predictions)
    tp = set(predictions) - set(fp)

    prec = len(tp) / (len(tp) + len(fp)) if (len(tp)+len(fp)) > 0 else 0
    rec = len(tp) / (len(tp) + len(fn)) if (len(tp)+len(fn)) > 0 else 0
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0

    return prec, rec, f1