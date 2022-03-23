import numpy as np
import torch

"""
custom_accuracy(predictions: tensor, target: tensor) -> local_accuracy: int
Input:
    - predictions: predicted labels
    - target: gold data
Output:
    - local_accuracy: portion of correct guessed labels
"""
def custom_accuracy(predictions, target):
    return len(np.intersect1d(predictions.numpy(), target.numpy()))