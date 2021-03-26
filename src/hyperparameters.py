"""
Function to compare hyperparameters.
"""
import numpy as np
from nn_regression import create_datasets, normalize, build_model, convergence_time
import matplotlib.pyplot as plt
from launch_barrier import pash_to_dataframe
from analysis import calculate_mse

def mse_test(model, train_features, train_target, test_features, test_target, epoch_no, batch_size, epoch_per_fit=10):
    """
    Takes a model, training datasets, test datasets, number of epochs and a batch size
    and returns a list of losses on training and test dataset calculated every epoch_per_fit epochs.
    :param model: (keras model) compiled model
    :param train_features: (pandas DataFrame) training features
    :param train_target: (pandas DataFrame) training targets
    :param test_features: (pandas DataFrame) test features
    :param test_target: (pandas DataFrame) test target
    :param epoch_no: (int) total number of epochs
    :param batch_size: (int) batch_size parameter for fit
    :param epoch_per_fit: (int) number of epochs for every fit at which point we obtain losses - 10 by default
    :return losses: (float list) list of losses on training set
    :return losses_test: (float list) list of losses on test set
    """
    losses = []
    losses_test = []
    for i in range(epoch_no//epoch_per_fit):
        history = model.fit(train_features, train_target, epochs=epoch_per_fit, batch_size=batch_size, verbose=0).history
        losses += history['loss']
        predicted_labels = np.ravel(model.predict(test_features))
        losses_test.append(calculate_mse(predicted_labels, test_target))
    return losses, losses_test


def hyper_analysis(dataset, features, n_layers=3, n_neurons_per_layer=100, batch_size=10, n_epochs=2000,
                   activation='relu', optimizer='Adamax', loss='mean_squared_error', frac = 0.5):
    '''
    Takes dataset, keys of features and values for hyperparameters (takes a default value when no input given) and
    returns the loss on training set and the loss on test set per epoch
    and also the final loss reached as a function of the varied hyperparameters
    :param dataset: (pandas DataFrame) initial DataFrame
    :param features: (str list) list of keys of features
    :param n_layers: (default int or int list) list of different numbers of layers to be tested or 3 by default
    :param n_neurons_per_layer: (default int or int list) list of different numbers of neurons per layer to be tested
    or 100 by default
    :param batch_size: (default int or int list) list of different numbers of batch sizes to be tested or 10 by default
    :param n_epochs: (default int or int list) list of different numbers of epochs to be tested or 2000 by default
    :param activation: (default str or str list) list of different activation functions to be tested
    or 'relu' by default
    :param optimizer: (default str or str list) list of different optimizers to be tested or 'adamax' by default
    :param loss: (default str or str list) list of different loss functions to be tested
    or 'mean_squared_error' by default
    :param frac: (float) fraction of the data turned into training set - half by default
    :return losses_train_epoch: (float list) list of losses for training set per epoch
    :return losses_test_epoch: (float list) list of losses for test set per epoch
    :return losses_train_hp: (float list) list of final losses for training set for different values of a hyperparameter
    :return losses_test_hp: (float list) list of final losses for test set for different values of a hyperparameter
    '''

    'transform input parameters into arrays'
    input_parameters = [n_layers, n_neurons_per_layer, batch_size, n_epochs, activation, optimizer, loss]
    for i in range(len(input_parameters)):
        if not (type(input_parameters[i]) == list):
            input_parameters[i] = [input_parameters[i]]

    'make an array of all combinations of parameters'
    parameter_mesh = np.array(np.meshgrid(input_parameters[0], input_parameters[1], input_parameters[2],
                            input_parameters[3], input_parameters[4], input_parameters[5], input_parameters[6]))
    parameter_combinations_mesh = parameter_mesh.T.reshape(-1, 7)
    parameter_combinations_mesh = parameter_combinations_mesh.tolist()
    for para in parameter_combinations_mesh:
        para[0] = int(para[0])
        para[1] = int(para[1])
        para[2] = int(para[2])
        para[3] = int(para[3])
    print(parameter_combinations_mesh)
    'make input data ready to be used'
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=frac)
    print(len(train_features))
    'model-building with the combinations of parameters'
    normalizer = normalize(train_features)
    models = [0] * (len(parameter_combinations_mesh))
    losses_train_epoch = [0] * (len(parameter_combinations_mesh))
    losses_test_epoch = [0] * (len(parameter_combinations_mesh))
    losses_train_hp = [0] * (len(parameter_combinations_mesh))
    losses_test_hp = [0] * (len(parameter_combinations_mesh))
    for i in range(len(parameter_combinations_mesh)):
        layer = np.ones(parameter_combinations_mesh[i][0], dtype=int) * parameter_combinations_mesh[i][1]
        print("model "+str(i+1)+"/"+str(len(parameter_combinations_mesh)))
        models[i] = build_model(normalizer, layer,
                                activation=parameter_combinations_mesh[i][4],
                                optimizer=parameter_combinations_mesh[i][5],
                                loss=parameter_combinations_mesh[i][6])
        loss_train_epoch, loss_test_epoch = mse_test(models[i], train_features, train_labels,
                                                     test_features, test_labels,
                                                     parameter_combinations_mesh[i][3],
                                                     parameter_combinations_mesh[i][2])
        losses_train_epoch[i] = loss_train_epoch
        losses_test_epoch[i] = loss_test_epoch
        losses_train_hp[i] = loss_train_epoch[convergence_time(loss_train_epoch)]
        losses_test_hp[i] = loss_test_epoch[convergence_time(loss_test_epoch)]

    return losses_train_epoch, losses_test_epoch, losses_train_hp, losses_test_hp

if __name__ == "__main__":
    # Import data
    dataset = pash_to_dataframe("data/pash/large_pash.dat")
    features = ["epsilon", "a3"]

    # Analysis of the different optimizers
    loss_train_epoch, loss_test_epoch, loss_train_hp, loss_test_hp = hyper_analysis(
        dataset, features,
        n_neurons_per_layer=150,
        n_layers=3,
        n_epochs=2000,
        frac=0.1,
        optimizer=['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])

    #plot the analysis
    opt = [1, 2, 3, 4, 5, 6, 7, 8]
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

    plt.plot(opt, loss_train_hp, 'g+')
    plt.plot(opt, loss_test_hp, 'b+')
    plt.xticks(opt, optimizer)
    plt.title("Loss as a function of different optimizers")
    plt.ylabel("Loss at the end of training")
    plt.xlabel("Optimizers")

    plt.show()
