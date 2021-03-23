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
n_layers = [1,3,5,10]
loss_train_epoch, loss_test_epoch, loss_train_hp, loss_test_hp = hyper_analysis(
    dataset, features, n_layers=n_layers, n_epochs=20)
epochs_train = np.arange(1, 21, 1)
epochs_test = np.arange(1, 21, 10)
fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
plt.figure()
plt.plot(n_layers, loss_train_hp, 'g+')
plt.plot(n_layers, loss_test_hp, 'b+')
ax1.plot(epochs_train, loss_train_epoch[0], 'b+')
ax1.plot(epochs_test, loss_test_epoch[0], 'r+')
ax2.plot(epochs_train, loss_train_epoch[1], 'b+')
ax2.plot(epochs_test, loss_test_epoch[1], 'r+')
ax3.plot(epochs_train, loss_train_epoch[2], 'b+')
ax3.plot(epochs_test, loss_test_epoch[2], 'r+')
ax4.plot(epochs_train, loss_train_epoch[3], 'b+')
ax4.plot(epochs_test, loss_test_epoch[3], 'r+')



plt.show()
#print(rmse)
