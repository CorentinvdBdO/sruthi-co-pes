import tensorflow as tf
import numpy as np
import math
from extract_pash import pash_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from launch_barrier import launch_barrier, input_template
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform

def calculate_rmse(predicted_values, real_values):
    """
    Takes dataframes to calculate root mean squared error
    :param predicted_values: dataframe
    :param real_values: dataframe
    :return: float
    """
    no_points = len(predicted_values)
    error = predicted_values-real_values
    rmse = math.sqrt(error*error/no_points)
    return rmse

def rmse_test(model, train_features, train_labels, test_features, test_labels, epoch_no, n_batch):
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
        history = model.fit(train_features, train_labels, epochs=epoch_no).history
        losses.append(history['loss'])
        predicted_labels = model.predict(test_features)
        losses_test.append(calculate_rmse(predicted_labels, test_labels))
    return losses, losses_test

def hyperparameter_analysis(dataset, n_layers=3, n_neurons_per_layer=100, n_batch=10, n_epochs=2000, activation='relu', optimizer='adam'):
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
    features = ["P(1)", "P(2)"]
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.1)
    normalizer = normalize(train_features)
    models = []
    losses = []
    losses_test = []
    default_layers = []
    for i in range(n_layers):
        default_layers.append(n_neurons_per_layer)
    default_model = build_model(normalizer, default_layers, )

    if type(n_layers) == 'list':
        layers = []
        histories = []
        for i in range(len(n_layers)):
            layers.append((np.ones(n_layers[i])*n_neurons_per_layer).tolist())
            models.append(build_model(normalizer, layers[i], n_batch))
            loss, loss_test = rmse_test(models[i], train_features, train_labels, test_features, test_labels, n_epochs)
            losses.append(loss)
            losses_test.append(loss_test)
    elif type(n_neurons_per_layer) == 'list':
        layers = []
        for i in range(n_layers):
            layers.append((np.ones(n_layers)*n_neurons_per_layer[i]).tolist())
            models.append(build_model(normalizer, layers[i], n_batch))
            loss, loss_test = rmse_test(models[i], train_features, train_labels, test_features, test_labels, n_epochs)
            losses.append(loss)
            losses_test.append(loss_test)
    elif type(n_batch) == 'list':
        for i in range(len(n_batch)):
            models.append(build_model(normalizer, default_layers, n_batch[i]))
            loss, loss_test = rmse_test(models[i], train_features, train_labels, test_features, test_labels, n_epochs)
            losses.append(loss)
            losses_test.append(loss_test)
    elif type(n_epochs)=='list':
        for i in range(len(n_epochs)):







    return min_losses



