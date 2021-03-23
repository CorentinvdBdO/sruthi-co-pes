import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from launch_barrier import launch_barrier, input_template
from extract_pash import pash_to_dataframe
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
from hyperparameters import calculate_mse, mse_test, hyper_analysis

dataset = pash_to_dataframe("barrier/pash_step3.dat")
features = ["epsilon", "a3"]

# train_dataset, test_dataset, \
#         train_features, train_labels, \
#          test_features, test_labels \
#         = create_datasets(dataset, features, "Barrier", frac=0.1)
# normalizer = normalize(train_features)
# model = build_model(normalizer, [100, 100, 100])

'mse calculation'
# epoch_no = 1000
# history = model.fit(train_features, train_labels, epochs=epoch_no).history
# predicted_labels = np.ravel(model.predict(dataset[features]))
# expected_labels = dataset.pop('Barrier')
# mse = calculate_mse(predicted_labels, expected_labels)

'mse test values vs train values'
# losses, losses_test = mse_test(model, train_features, train_labels, test_features, test_labels, 200, len(train_features))
# epochs1 = np.arange(1, 2001, 1)
# epochs2 = np.arange(1, 2001, 10)
# plt.plot(epochs1, losses, 'bo')
# plt.plot(epochs2, losses_test, 'go')
# plt.yscale("log")

'hyper_analysis tests'
loss_train_epoch, loss_test_epoch, loss_train_hp, loss_test_hp = hyper_analysis(
    dataset, features,
    n_neurons_per_layer=[5, 10, 20, 40, 70, 100, 150, 200, 300, 400, 500],
    n_layers=3,
    n_epochs=1500,
    frac=0.1)
n_neurons_per_layer=[5, 10, 20, 40, 70, 100, 150, 200, 300, 400, 500]

plt.plot(n_neurons_per_layer, loss_train_hp, 'g+')
plt.plot(n_neurons_per_layer, loss_test_hp, 'b+')

plt.title("Loss as a function of the number of neurons")
plt.ylabel("Loss at the end of training")
plt.xlabel("Number of neurons")

plt.show()



plt.show()
#print(rmse)
