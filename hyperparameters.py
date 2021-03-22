import tensorflow as tf
import numpy as np
import math
from extract_pash import pash_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from launch_barrier import launch_barrier, input_template

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

def rmse_test(model, train_features, train_labels, test_features, test_labels, epoch_no):
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
    return losses_test

