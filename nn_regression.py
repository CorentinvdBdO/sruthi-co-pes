import tensorflow as tf
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface
import matplotlib.pyplot as plt
import pandas as pd

def dataframe_to_dataset(dataframe, features, targets):
    """
    Turns a pandas dataframe into a tensorflow dataset
    :param dataframe: pandas dataframe
    :param features: str or list of str out of dataframe's columns
    :param targets: str or list of str out of dataframe's columns
    :return: dataset
    """
    dataset = (tf.data.Dataset.from_tensor_slices((
        dataframe[features].values,
        dataframe[targets].values)))
    return dataset

def create_datasets(data, features, target, frac=0.5):
    """
    Takes full data and keys of interest to return training and testing data
    :param data: Input DataFrame
    :param features: str or list f str keys of features
    :param target: str or list f str keys of tragets
    :param frac: fraction of the data turned into training
    :return: DataFrames: full training and testing sets,
                        and then separated by features and target
    """
    data = data[features + [target]]
    train_data = data.sample(frac=frac, random_state=0)
    test_data = data.drop(train_data.index)
    train_features = train_data.copy()
    test_features = test_data.copy()
    train_target = train_features.pop(target)
    test_target = test_features.pop(target)
    return train_data, test_data, train_features, train_target, test_features,test_target

def normalize(data):
    """
    Returns a Keras normalizer adapted to the input features data
    :param data: The features DataFrame to normalize
    :return normalizer: normalizer object
    """
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(np.array(data))
    return normalizer

def build_model(normalizer, layers, input_dim = 2, activation='relu', optimizer='adam'):
    """
    Takes a normalizer adapted to input and the form of neural network
            and return a compiled keras model
    :param normalizer: normalizer
    :param layers: int list [m,n,..] meaning a first hidden layer with m neurones,
    a second hidden layer with n neurones, etc.
    :param input_dim: number of features
    :param activation: neuron's activation function
    :param optimizer: optimizer used to compile
    :return model: compiled keras model
    """
    model = tf.keras.Sequential([normalizer])
    for i in range(len(layers)):
        neurons_no = layers[i]
        # The first layers need to know the number of inputs
        if (i==0):
            model.add(tf.keras.layers.Dense(neurons_no, input_dim=input_dim, activation=activation ))
        else:
            model.add(tf.keras.layers.Dense(neurons_no, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def learning_curve(history):
    """
    plot Loss = f(epochs)
    :param losses: History.history, output of a fit.
    """
    losses = history['loss']
    plt.plot(losses)

def convergence_time(history, threshold = 4, req_time_confined = 10):
    """
    returns the start epoch from which the loss has been within threshold
    of the last observed loss for ta least 10 epochs.
    else, return the last epoch
    Considers that the epochs range from (0, len(losses))
    """
    losses = history['loss']
    for i in range(len(losses)):
        rest = losses[i:]
        rest = np.mean(rest)
        if losses[i] < rest:
            plt.vlines(i, 0, 10, colors="r")
            return i


def retransform(data, predicted_target, target_keys=["Barrier"]):
    """
    makes one dataframe from the features and the prediction
    :param data: a dataframe
    :param predicted_target: array containing the predicted target
    :return: result of the concatenation
    """
    retransformed_data = data.join(pd.DataFrame(predicted_target, columns=target_keys))
    return retransformed_data

def plot_surface_diff(data1,data2, key1, key2, key3, ax = None):
    """
    Plot the surface difference between key3 of dataframes 1 & 2
    :param data1: dataframe
    :param data2: dataframe
    :param key1: x axis
    :param key2: y axis
    :param key3: z axis of the graph, on which the difference is made
    :param ax: on which the surface is plotted
    """
    diff = data1[key3] - data2[key3]
    data_diff = retransform(data1[[key1, key2]], diff)
    plot_surface(data_diff, key1, key2, key3, ax)

if __name__ == "__main__":
    # Get the data
    data =pash_to_dataframe("barrier/pash_step3new.dat")
    train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(data, ["epsilon", "a3"], "Barrier", frac=0.5)
    print(train_dataset, data, test_dataset)
    normalizer = normalize(train_features)
    model = build_model(normalizer, [100, 100, 100])
    history = model.fit(train_features, train_labels, epochs=5000).history
    print(convergence_time(history))
    learning_curve(history)
    plt.yscale("log")

    plt.ylabel("MSE")
    plt.xlabel("epoch")
    plt.show()
    plt.plot(test_labels, model.predict(test_features), '.')
    plt.show()
