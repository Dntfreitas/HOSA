import numpy as np
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, roc_auc_score
from sklearn.utils.multiclass import unique_labels


def metrics_multilabel(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    tp = confusion_matrices[:, 0, 0]
    fn = confusion_matrices[:, 0, 1]
    fp = confusion_matrices[:, 1, 0]
    tn = confusion_matrices[:, 1, 1]
    sensitivity = np.mean(tp / (tp + fn))
    specificity = np.mean(tn / (fp + tn))
    accuracy = accuracy_score(y_true, y_pred)
    if len(unique_labels(y_true)) == 2:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc_value = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
    return auc_value, accuracy, sensitivity, specificity
