"""
Functions used to manipulate DataFrames for a keras fit, to build models quickly and work on the learning curves.
"""
import tensorflow as tf
import numpy as np
from launch_barrier import pash_to_dataframe
import matplotlib.pyplot as plt
import pandas as pd


def create_datasets(data, features, target, frac=0.5):
    """
    Takes full DataFrame directly transformed from a pash.dat file
    and keys of interest to return training and testing datasets.
    :param data: (pandas DataFrame) DataFrame of a pash.dat file
    :param features: (str list) keys of features
    :param target: (str) key of target
    :param frac: (float) fraction of the data turned into training set - half by default
    :return train_data: (pandas DataFrames) training set with columns corresponding to features and target
    :return test_data: (pandas DataFrames) test set with columns corresponding to features and target
    :return train_features: (pandas DataFrames) training set with columns corresponding to features
    :return train_target: (pandas DataFrames) training set with column corresponding to target
    :return test_features: (pandas DataFrames) test set with columns corresponding to features
    :return test_target: (pandas DataFrames) test set with column corresponding to target
    """
    data = data[features + [target]]
    if type(frac) is int:
        train_data = data.sample(n=frac, random_state=0)
    else:
        train_data = data.sample(frac=frac, random_state=0)
    test_data = data.drop(train_data.index)
    train_features = train_data.copy()
    test_features = test_data.copy()
    train_target = train_features.pop(target)
    test_target = test_features.pop(target)
    return train_data, test_data, train_features, train_target, test_features, test_target


def normalize(data):
    """
    Takes a DataFrame and returns an adapted Keras normalizer.
    :param data: (pandas DataFrame) features DataFrame to normalize
    :return normalizer: (keras normalizer object) normalizer
    """
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(np.array(data))
    return normalizer

def build_model(normalizer, layers, input_dim = 2, activation='relu', optimizer='adam',loss='mean_squared_error'):
    """
    Takes a normalizer adapted to input, the form of neural network, the dimension of input
    and the choice of activation function, optimizer and loss function.
    Returns a compiled keras model.
    :param normalizer: (keras normalizer object) normalizer
    :param layers: (int list) [m,n,..] meaning a first hidden layer with m neurones, a second with n neurones, etc.
    :param input_dim: (int) number of features
    :param activation: (str) neuron's activation function
    :param optimizer: (str) optimizer used to compile
    :return model: (keras model) compiled model
    """
    model = tf.keras.Sequential([normalizer])
    for i in range(len(layers)):
        neurons_no = layers[i]
        # The first layer needs to know the number of inputs
        if (i==0):
            model.add(tf.keras.layers.Dense(neurons_no, input_dim=input_dim, activation=activation ))
        else:
            model.add(tf.keras.layers.Dense(neurons_no, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def learning_curve(losses, ax=plt.gca(), log=False, y="Loss", title="Learning curve and convergence time"):
    """
    Takes the list of losses and plots it as a function of epochs with a vertical line at convergence.
    :param losses: (float list) History.history['loss'] from output of a fit
    :param ax: (axes) plt.gca() by default
    :param log: (bool) If True the plot is in log scale
    :param title: (str) Title of the plot
    """
    ax.plot(losses, "+")
    ax.set_ylabel(title)
    ax.set_xlabel("epoch")
    ax.set_title(title)
    time = convergence_time(losses)
    plt.vlines(time, 0, max(losses), colors="r")
    if log:
        ax.set_yscale("log")

def convergence_time(losses):
    """
    Takes a list of losses and returns the start epoch from which the loss has been smaller or equal to the
    mean loss at the end for at least 10 epochs or simply returns the last epoch.
    Considers that the epochs range from (0, len(losses)).
    :param losses: (float list) History.history['loss'] from output of a fit
    """
    for i in range(len(losses)):
        rest = losses[i:]
        rest = np.mean(rest)
        if losses[i] <= rest:
            return i
        return len(losses)-1


def retransform(data, predicted_target, target_keys=["Barrier"]):
    """
    Takes a DataFrame and a single-columned DataFrame added to the former with the given key.
    :param data: (pandas DataFrame) DataFrame to be modified
    :param predicted_target: (pandas DataFrame) DataFrame with values to be added to data
    :param target_keys: (str list) key of column to be added
    :return retransformed_data: (pandas DataFrame) DataFrame after concatenation of column
    """
    retransformed_data = data.join(pd.DataFrame(predicted_target, columns=target_keys))
    return retransformed_data



if __name__ == "__main__":
    # Get data
    data =pash_to_dataframe("data/pash/large_pash.dat")
    train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(data, ["epsilon", "a3"], "Barrier", frac=0.1)
    # Build model
    normalizer = normalize(train_features)
    model = build_model(normalizer, [150, 150, 150], optimizer="adamax")
    # Fit model
    losses = model.fit(train_features, train_labels, epochs=1000).history['loss']
    # Save model
    model.save("data/models/example_NN")
    # Plot the learning curve
    learning_curve(losses, y="MSE", log= not (0 in losses))
    plt.show()

