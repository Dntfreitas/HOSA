import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, balanced_accuracy_score


def metrics_multiclass(y_true, y_probs, n_classes, inbalance_correction=False):
    y_pred = np.argmax(y_probs, axis=1)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]
    if inbalance_correction:
        classes_weight = np.sum(mcm[:, 1, :], axis=1) / np.sum(mcm[:, 1, :])
        sensitivity = np.average(tp / (tp + fn), weights=classes_weight)
        specificity = np.average(tn / (fp + tn), weights=classes_weight)
        accuracy = balanced_accuracy_score(y_true, y_pred)
    else:
        sensitivity = np.mean(tp / (tp + fn))
        specificity = np.mean(tn / (fp + tn))
        accuracy = accuracy_score(y_true, y_pred)
    if n_classes > 2:
        if inbalance_correction:
            auc_value = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        else:
            auc_value = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
    else:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    return auc_value, accuracy, sensitivity, specificity
