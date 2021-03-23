import matplotlib.pyplot as plt
import numpy as np
from extract_pash import pash_to_dataframe
from hyperparameters import hyper_analysis

dataset = pash_to_dataframe("barrier/large_pash.dat")
features = ["epsilon", "a3"]

'hyper_analysis tests'
loss_train_epoch, loss_test_epoch, loss_train_hp, loss_test_hp = hyper_analysis(
    dataset, features,
    n_neurons_per_layer=150,
    n_layers=3,
    n_epochs=1500,
    frac=0.1,
    optimizer=['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])

opt=[1,2,3,4,5,6,7,8]
optimizer=['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
#
plt.plot(opt, loss_train_hp, 'g+')
plt.plot(opt, loss_test_hp, 'b+')
plt.xticks(opt, optimizer)
# n_epochs = 1500
# epochs1=np.arange(1,n_epochs+1,1)
# epochs2=np.arange(1,n_epochs+1,10)
# plt.plot(epochs1, loss_train_epoch[0],'g+')
# plt.plot(epochs2, loss_test_epoch[0],'b+')
plt.title("Loss as a function of different optimizers")
plt.ylabel("Loss at the end of training")
plt.xlabel("Optimizers")



plt.show()
