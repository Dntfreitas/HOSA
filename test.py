import pandas as pd

from hosa.Models.CNN import CNNRegression
from hosa.Optimization import HOSA

# 1 - Download, load, and split the data
dataset = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', header=0, index_col=0)
X = dataset.Passengers.to_numpy().reshape((len(dataset), 1))
y = dataset.Passengers.to_numpy()
# 2 - Specify the parameters' values to test
param_grid = {
        'n_neurons_dense_layer': [5, 10],
        'gol_sizes':             [[3], [4], [5]],
        'overlapping_type':      ['left', 'central', 'right'],
        'overlapping_epochs':    [3],
        'stride':                [1],
}
# 3 - Create a HOSA instance and find the best set of parameters
clf = HOSA(X, y, CNNRegression, 1, param_grid, 0.1, apply_rsv=True, n_splits=3)
clf.fit(verbose=0)
# 4 - Save the best model
best_parms = clf.get_params()
best_model = clf.get_model()
best_model.save('saved_model/my_model')