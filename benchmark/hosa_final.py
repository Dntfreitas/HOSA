import pandas as pd

from utils import *

# Start HOSA
n_outputs = 2
n_neurons_dense_layer = 200
n_units = 400
n_subs_layers = 1
is_bidirectional = True
model_type = 'lstm'
optimizer = 'adam'
dropout_percentage = 0.1
metrics = ['accuracy']
activation_function_dense = 'relu'
kernel_initializer = 'normal'
batch_size = 1000
epochs = 50
patience = 5
timesteps = 15
overlapping_epochs = 0
overlapping_type = 'left'


def train_test(x_tr, y_tr, x_te, y_te):
    model = RNNClassification(n_outputs=n_outputs, n_neurons_dense_layer=n_neurons_dense_layer, n_units=n_units, n_subs_layers=n_subs_layers, is_bidirectional=is_bidirectional, model_type=model_type, optimizer=optimizer, dropout_percentage=dropout_percentage, metrics=metrics, activation_function_dense=activation_function_dense, kernel_initializer=kernel_initializer, batch_size=batch_size, epochs=epochs, patience=patience)
    x_win_train, y_win_train = create_overlapping(x_tr, y_tr, model, overlapping_epochs, overlapping_type, n_stride=1, n_timesteps=timesteps)
    x_win_test, y_win_test = create_overlapping(x_te, y_te, model, overlapping_epochs, overlapping_type, n_stride=1, n_timesteps=timesteps)
    model.prepare(x_win_train, y_win_train)
    model.compile()
    model.fit(x_win_train, y_win_train)
    auc_value, accuracy, sensitivity, specificity = model.score(x_win_test, y_win_test)
    return auc_value, accuracy, sensitivity, specificity


x_train, y_train, x_test, y_test = prepare_data()

auc_value_f1, accuracy_f1, sensitivity_f1, specificity_f1 = train_test(x_train, y_train, x_test, y_test)
auc_value_f2, accuracy_f2, sensitivity_f2, specificity_f2 = train_test(x_test, y_test, x_train, y_train)
auc_value = (auc_value_f1 + auc_value_f2) / 2
accuracy = (accuracy_f1 + accuracy_f2) / 2
sensitivity = (sensitivity_f1 + sensitivity_f2) / 2
specificity = (specificity_f1 + specificity_f2) / 2

results = pd.DataFrame(
        {'auc_value': auc_value, 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'auc_value_f1': auc_value_f1, 'accuracy_f1': accuracy_f1, 'sensitivity_f1': sensitivity_f1, 'specificity_f1': specificity_f1, 'auc_value_f2': auc_value_f2, 'accuracy_f2': accuracy_f2, 'sensitivity_f2': sensitivity_f2, 'specificity_f2': specificity_f2},
        index=[0]
)
results.to_csv('logs/hosa_final_dataset.csv', index=False)
