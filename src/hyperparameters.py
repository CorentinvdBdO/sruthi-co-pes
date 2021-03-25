import numpy as np
from nn_regression import create_datasets, normalize, build_model, convergence_time
import matplotlib.pyplot as plt
from launch_barrier import pash_to_dataframe

def calculate_mse(predicted_values, real_values):
    """
    Takes dataframes to calculate root mean squared error
    :param predicted_values: dataframe
    :param real_values: dataframe
    :return: float
    """
    no_points = len(predicted_values)
    error = predicted_values - real_values
    mse = np.sum(error * error) / no_points
    return mse


def mse_test(model, train_features, train_labels, test_features, test_labels, epoch_no, n_batch):
    """
    Takes training datasets, test datasets and number of epochs to give list of losses on test dataset per epoch
    :param model: keras model
    :param train_features: dataframe
    :param train_labels: dataframe
    :param test_features: dataframe
    :param test_labels: dataframe
    :param epoch_no: int
    :return: list
    """
    losses = []
    losses_test = []
    epoch_per_fit = 10
    for i in range(epoch_no//epoch_per_fit):
        history = model.fit(train_features, train_labels, epochs=epoch_per_fit, batch_size=n_batch, verbose=0).history
        losses += history['loss']
        predicted_labels = np.ravel(model.predict(test_features))
        losses_test.append(calculate_mse(predicted_labels, test_labels))
    return losses, losses_test

def hyper_analysis(dataset, features, n_layers=3, n_neurons_per_layer=100, batchsize=10, n_epochs=2000,
                   activation='relu', optimizer='adam', loss='mean_squared_error', frac = 0.5):
    '''
    Takes dataset, features and a list of values for hyperparameters (takes a default value when no input given) and
    returns the loss on training set and the loss on test set per epoch
    and also the final loss reached as a function of the varied hyperparameters
    :param dataset: dataset
    :param features: dataset
    :param n_layers: default int or int list
    :param n_neurons_per_layer: default int or int list
    :param batchsize: default int or int list
    :param n_epochs: default int or int list
    :param activation: default activation function or string list
    :param optimizer: default optimizer or string list
    :return: 4 lists
    '''

    'transform input parameters into arrays'
    input_parameters = [n_layers, n_neurons_per_layer, batchsize, n_epochs, activation, optimizer, loss]
    for i in range(len(input_parameters)):
        if not (type(input_parameters[i]) == list):
            input_parameters[i] = [input_parameters[i]]

    'make an array of all combinations of parameters'
    parameter_mesh = np.array(np.meshgrid(input_parameters[0], input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4],
                                          input_parameters[5], input_parameters[6]))
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
