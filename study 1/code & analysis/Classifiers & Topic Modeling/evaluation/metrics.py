import numpy as np
from sklearn.metrics import hamming_loss, precision_recall_fscore_support, accuracy_score


def compute_binary_classification_metrics(true_labels, predicted_labels):
    """Compute precision, recall, F1 score, and accuracy for binary classification"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary"
    )
    accuracy = (np.array(predicted_labels) == np.array(true_labels)).mean()
    return precision, recall, f1, accuracy


def compute_multiclass_classification_metrics(true_labels, predicted_labels):
    """
    Compute precision, recall, F1 score, and accuracy for multiclass classification

    Parameters:
    true_labels (list/array): True labels
    predicted_labels (list/array): Predicted labels
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted"
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="micro"
    )
    accuracy = accuracy_score(true_labels, predicted_labels)
    return (
        precision_macro,
        recall_macro,
        f1_macro,
        precision_micro,
        recall_micro,
        f1_micro,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        accuracy,
    )


def compute_multilabel_classification_metrics(true_labels, predicted_labels):
    """Compute precision, recall, F1 score, and accuracy for multilabel classification"""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted"
    )
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="samples"
    )
    hamming = hamming_loss(true_labels, predicted_labels)
    accuracy = (np.array(predicted_labels) == np.array(true_labels)).mean()
    return (
        precision_macro,
        recall_macro,
        f1_macro,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        precision_samples,
        recall_samples,
        f1_samples,
        hamming,
        accuracy,
    )
