import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical

from CNNClassification import CNNClassification
from create_overlapping import create_overlapping


def hosa_cnn(X, y, g_max, o_max, n_start, n_step, n_max, m_start, m_max, mul_max, epsilon, batch_size=1000, epochs=50, patientece=5, verbose=1):
    G = np.arange(1, g_max + 1)
    O = np.arange(2, o_max + 1, 2)  # TODO: check when 0; should be only even?
    K = 2 ** np.arange(m_start, m_max + 1)
    N = np.arange(n_start, n_max + 1, n_step)
    MUL = np.arange(1, mul_max + 1)
    overlapping_types = ['central', 'left', 'right']
    n_classes = np.unique(y)

    best_auc_global = -np.inf
    best_model_global = None
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    auc_scores = []

    stop_training = False

    for g in G:  # Number of GofLayers
        if not stop_training:
            best_auc = -np.inf
            best_model = None
            for o in O:  # Overlapping duration
                for k in K:  # Number of kernels
                    for n in N:  # Number of neurons
                        for overlapping_type in overlapping_types:
                            for mul in MUL:
                                k_prev = np.nan
                                for i in range(g + 1):
                                    # Compute the overlapping
                                    X_new, y_new = create_overlapping(X, y, overlapping_type, o, stride=1)
                                    # Train/test split
                                    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33)
                                    # Compute class weights
                                    class_weights = class_weight.compute_class_weight('balanced', n_classes, y_train)
                                    class_weights = {j: class_weights[j] for j in range(len(n_classes))}
                                    # To categorical
                                    y_train = to_categorical(y_train)
                                    y_test = to_categorical(y_test)
                                    # Compute the GofLayer sizes
                                    gof_layer_sizes = []
                                    if i == 0:
                                        gof_layer_sizes.append(2 ** k)
                                        k_prev = 2 ** K
                                    else:
                                        gof_layer_sizes.append(k_prev * mul)
                                    # Compose and fit the model
                                    model = CNNClassification(class_weights, n, gof_layer_sizes, batch_size=batch_size, epochs=epochs, patientece=patientece, verbose=verbose)
                                    model.prepare(X_train.shape[1])
                                    model.compile()
                                    model.fit(X_train, y_train)
                                    # Test the model
                                    accuracy, sensitivity, specificity, auc = model.score(X_test, y_test)
                                    accuracy_scores.append(accuracy)
                                    sensitivity_scores.append(sensitivity)
                                    specificity_scores.append(specificity)
                                    auc_scores.append(auc)
                                    # Update current best AUC
                                    if auc > best_auc:
                                        best_auc = auc
                                        best_model = model
            if abs(best_auc - best_auc_global) <= epsilon:
                stop_training = True
            if best_auc > best_auc_global:
                best_auc_global = best_auc
                best_model_global = best_model
    return best_model_global
