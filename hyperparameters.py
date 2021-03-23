import tensorflow as tf
import numpy as np
import math
from extract_pash import pash_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from launch_barrier import launch_barrier, input_template
from nn_regression import create_datasets, normalize, build_model, convergence_time

def calculate_mse(predicted_values, real_values):
    """
    Takes dataframes to calculate root mean squared error
    :param predicted_values: dataframe
    :param real_values: dataframe
    :return: float
    """
    no_points = len(predicted_values)
    error = predicted_values-real_values
    mse = np.sum(error*error)/no_points
    return mse

def mse_test(model, train_features, train_labels, test_features, test_labels, epoch_no, n_batch):
    """
    Takes training datasets, test datasets and number of epochs to give list of losses on test dataset per epoch
    :param model: keras model
    :param train_features: dataset
    :param train_labels: dataset
    :param test_features: dataset
    :param test_labels: dataset
    :param epoch_no: int
    :return: list
    """
    losses = []
    losses_test = []
    for i in range(epoch_no):
        history = model.fit(train_features, train_labels, epochs=10, batch_size=n_batch, verbose=0).history
        losses+=history['loss']
        predicted_labels = np.ravel(model.predict(test_features))
        losses_test.append(calculate_mse(predicted_labels, test_labels))
    return losses, losses_test

def hyperparameter_analysis(dataset, features, n_layers=3, n_neurons_per_layer=100,
                            n_batch=10, n_epochs=200, activation='relu', optimizer='adam', loss='mean_squared_error'):
    """
    Takes the name of hyperparameter and a list with the values for this hyperparameter or
    a string list for activation functions an loss optimizers (default values for the rest)
    and returns the list with the minimum loss obtained for each value of the modified hyperparameter
    :param data: dataset
    :param n_layers: default or list
    :param n_neurons_per_layer: default or list
    :param n_batch: default or list
    :param n_epochs: default or list
    :param activation: default or str list
    :param optimizer: default or str list
    :return: float list
    """
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.1)
    normalizer = normalize(train_features)
    default_layers = []
    for i in range(n_layers):
        default_layers.append(n_neurons_per_layer)
    default_model = build_model(normalizer, default_layers)
    models = []
    losses = []
    losses_test = []


    if type(n_layers) == 'list':
        layers = []
        histories = []
        for i in range(len(n_layers)):
            layers.append((np.ones(n_layers[i])*n_neurons_per_layer).tolist())
            models.append(build_model(normalizer, layers[i], n_batch))
            loss, loss_test = mse_test(models[i], train_features, train_labels, test_features, test_labels, n_epochs, n_batch)
            losses.append(loss)
            losses_test.append(loss_test)

    elif type(n_neurons_per_layer) == 'list':
        layers = []
        for i in range(n_layers):
            layers.append((np.ones(n_layers)*n_neurons_per_layer[i]).tolist())
            models.append(build_model(normalizer, layers[i]))
            loss, loss_test = mse_test(models[i], train_features, train_labels, test_features, test_labels, n_epochs, n_batch)
            losses.append(loss)
            losses_test.append(loss_test)

    elif type(n_batch) == 'list':
        for i in range(len(n_batch)):
            loss, loss_test = mse_test(default_model, train_features, train_labels, test_features, test_labels, n_epochs, n_batch[i])
            losses.append(loss)
            losses_test.append(loss_test)

    elif type(n_epochs)=='list':
        for i in range(len(n_epochs)):
            loss, loss_test = mse_test(default_model, train_features, train_labels, test_features, test_labels,
                                        n_epochs[i], n_batch)
            losses.append(loss)
            losses_test.append(loss_test)

    elif type(activation)=='list':
        for i in range(len(activation)):
            models.append(build_model(normalizer, default_layers, activation=activation[i]))
            loss, loss_test = mse_test(default_model, train_features, train_labels, test_features, test_labels,
                                        n_epochs, n_batch)
            losses.append(loss)
            losses_test.append(loss_test)

    elif type(optimizer)=='list':
        for i in range(len(optimizer)):
            models.append(build_model(normalizer, default_layers, optimizer=optimizer[i]))
            loss, loss_test = mse_test(default_model, train_features, train_labels, test_features, test_labels,
                                        n_epochs, n_batch)
            losses.append(loss)
            losses_test.append(loss_test)

    return losses, losses_test

def hyper_analysis(dataset, features, n_layers=3, n_neurons_per_layer=100, batchsize=10, n_epochs=200,
                   activation='relu', optimizer='adam', loss='mean_squared_error'):
    '''

    :param dataset:
    :param features:
    :param n_layers:
    :param n_neurons_per_layer:
    :param batchsize:
    :param n_epochs:
    :param activation:
    :param optimizer:
    :return:
    '''

    'transform input parameters into arrays'
    input_parameters = [n_layers, n_neurons_per_layer, batchsize, n_epochs, activation, optimizer, loss]
    for i in range(len(input_parameters)):
        if not(type(input_parameters[i])==list):
            input_parameters[i] = [input_parameters[i]]

    'layer structure'
    layer_mesh = np.array(np.meshgrid(input_parameters[0], input_parameters[1])).T.reshape(-1, 2)
    layers = [0]*len(layer_mesh)
    for i in range(len(layer_mesh)):
        layers[i] = np.ones(layer_mesh[i][0]) * layer_mesh[i][1]

    'make an array of all combinations of parameters'
    parameter_mesh = np.array(np.meshgrid(layers, input_parameters[2], input_parameters[3], input_parameters[4],
                                          input_parameters[5], input_parameters[6]))
    parameter_combinations_mesh = parameter_mesh.T.reshape(-1,6)

    'make input data ready to be used'
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.1)

    'model-building with the combinations of parameters'
    normalizer = normalize(train_features)
    models = [0]*(len(parameter_combinations_mesh))
    losses_train_epoch = [0]*(len(parameter_combinations_mesh))
    losses_test_epoch = [0]*(len(parameter_combinations_mesh))
    losses_train_hp = [0]*(len(parameter_combinations_mesh))
    losses_test_hp = [0]*(len(parameter_combinations_mesh))
    for i in range(len(parameter_combinations_mesh)):
        models[i] = build_model(normalizer, parameter_combinations_mesh[i][0],
                                activation=parameter_combinations_mesh[i][3],
                                optimizer=parameter_combinations_mesh[i][4],
                                loss=parameter_combinations_mesh[i][5])
        loss_train_epoch, loss_test_epoch = mse_test(models[i], train_features, train_labels,
                                                         test_features, test_labels,
                                                         parameter_combinations_mesh[i][2],
                                                         parameter_combinations_mesh[i][1])

        losses_train_epoch[i] = loss_train_epoch
        losses_test_epoch[i] = loss_test_epoch
        losses_train_hp[i] = loss_train_epoch[convergence_time(loss_train_epoch)]
        losses_test_hp[i] = loss_test_epoch[convergence_time(loss_test_epoch)]


    return losses_train_epoch, losses_test_epoch, losses_train_hp, losses_test_hp







