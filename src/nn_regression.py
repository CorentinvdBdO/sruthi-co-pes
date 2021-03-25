import shutil
import tensorflow as tf
import numpy as np
from launch_barrier import pash_to_dataframe
import matplotlib.pyplot as plt
import pandas as pd

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
    if type(frac) is int:
        train_data = data.sample(n=frac, random_state=0)
    else:
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

def build_model(normalizer, layers, input_dim = 2, activation='relu', optimizer='adam',loss='mean_squared_error'):
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
        # The first layer need to know the number of inputs
        if (i==0):
            model.add(tf.keras.layers.Dense(neurons_no, input_dim=input_dim, activation=activation ))
        else:
            model.add(tf.keras.layers.Dense(neurons_no, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def learning_curve(losses, ax=plt.gca(), log=False, y="Loss", title="Learning curve and convergence time"):
    """
    plot Loss = f(epochs)
    :param losses: History.history, output of a fit.
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
    returns the start epoch from which the loss has been within threshold
    of the last observed loss for ta least 10 epochs.
    else, return the last epoch
    Considers that the epochs range from (0, len(losses))
    """
    for i in range(len(losses)):
        rest = losses[i:]
        rest = np.mean(rest)
        if losses[i] <= rest:
            return i
        return len(losses)-1


def retransform(data, predicted_target, target_keys=["Barrier"]):
    """
    makes one dataframe from the features and the prediction
    :param data: a dataframe
    :param predicted_target: array containing the predicted target
    :return: result of the concatenation
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

