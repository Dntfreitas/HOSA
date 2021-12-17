import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score
from sklearn.utils import compute_class_weight


def metrics_multiclass(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = confusion_matrices[:, 0, 0], confusion_matrices[:, 0, 1], confusion_matrices[:, 1, 0], confusion_matrices[:, 1, 1]
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_true), y=y_true)
    sensitivity = np.average(tp / (tp + fn), weights=class_weight)
    specificity = np.average(tn / (fp + tn), weights=class_weight)
    accuracy = np.average((tp + tn) / (tp + tn + fn + fp), weights=class_weight)
    if len(class_weight) <= 2:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc_value = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
    return auc_value, accuracy, sensitivity, specificity
