import matplotlib.pyplot as plt
from extract_pash import pash_to_dataframe
from hyperparameters import hyper_analysis

dataset = pash_to_dataframe("barrier/large_pash.dat")
features = ["epsilon", "a3"]

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
