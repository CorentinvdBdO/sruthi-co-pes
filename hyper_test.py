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

dataset = pash_to_dataframe("barrier/large_pash.dat")
features = ["epsilon", "a3"]

train_dataset, test_dataset, \
        train_features, train_labels, \
         test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.1)
for i in train_features.index:
    print(train_features.loc[i,"epsilon"])
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

# 'hyper_analysis tests'
# loss_train_epoch, loss_test_epoch, loss_train_hp, loss_test_hp = hyper_analysis(
#     dataset, features,
#     n_neurons_per_layer=150,
#     n_layers=3,
#     n_epochs=1500,
#     frac=0.1,
#     optimizer=['RMSprop', 'Adam', 'Adamax', 'Nadam'])
#
# opt=[1,2,3,4]
# optimizer=['RMSprop', 'Adam', 'Adamax', 'Nadam']
#
# # los=[1,2,3,4,5]
# # loss=['mean_squared_error', 'mean_absolute_error', 'cosine_similarity', 'huber', 'logcosh']
#
# # plt.plot(opt, loss_train_hp, 'g+')
# # plt.plot(opt, loss_test_hp, 'b+')
# # plt.xticks(opt, optimizer)
# n_epochs = 1500
# epochs1=np.arange(1,n_epochs+1,1)
# epochs2=np.arange(1,n_epochs+1,10)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# fig.suptitle("Learning curves for different optimizers")
# plt.yscale("log")
#
# ax1.semilogy(epochs1, loss_train_epoch[0],'g+')
# ax1.semilogy(epochs2, loss_test_epoch[0],'b+')
# ax1.set_ylabel("Loss")
# ax1.set_title("RMSprop")
#
# ax2.semilogy(epochs1, loss_train_epoch[1],'g+')
# ax2.semilogy(epochs2, loss_test_epoch[1],'b+')
# ax2.set_ylabel("Loss")
# ax2.set_title("Adam")
#
# ax3.semilogy(epochs1, loss_train_epoch[2],'g+')
# ax3.semilogy(epochs2, loss_test_epoch[2],'b+')
# ax3.set_xlabel("Number of epochs")
# ax3.set_ylabel("Loss")
# ax3.set_title("Adamax")
#
# ax4.semilogy(epochs1, loss_train_epoch[3],'g+')
# ax4.semilogy(epochs2, loss_test_epoch[3],'b+')
# ax4.set_xlabel("Number of epochs")
# ax4.set_ylabel("Loss")
# ax4.set_title("Nadam")
#
# plt.show()



