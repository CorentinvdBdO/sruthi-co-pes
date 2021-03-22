import tensorflow as tf
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from launch_barrier import launch_barrier, input_template

def dataframe_to_dataset(dataframe, features_key, targets_key):
    """
    Turns a pandas dataframe into a tensorflow dataset
    :param dataframe: pandas dataframe
    :param features: str or list of str out of dataframe's columns
    :param targets: str or list of str out of dataframe's columns
    :return: dataset
    """
    dataset = (tf.data.Dataset.from_tensor_slices((
        dataframe[features_key].values,
        dataframe[targets_key].values)))
    return dataset

def dataset_to_dataframe(dataset, features_key, labels_key):
    return 0

def create_datasets(data, features, target, frac=0.5):
    """
    Takes a file and keys of interest to return two datasets - one for training and one for testing
    :param path: path to file with the output data of BARRIER
    :param feature_key_1: str key of first parameter x
    :param feature_key_2: str key of second parameter y
    :param target_key: str key of quantity of interest z = f(x,y)
    :return: dataset consisting of columns given by x, y and z seperated randomly into a training dataset
    and a test dataset
    """
    dataset = data[features + [target]]
    train_dataset = dataset.sample(frac=frac, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop(target)
    test_labels = test_features.pop(target)
    return train_dataset, test_dataset, train_features, train_labels, test_features,test_labels

def normalize(train_features):
    """
    Returns a Keras normalizer adapted to the dataset
    :param dataset: dataset
    :return: normalizer
    """
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    return normalizer

def build_model(normalizer, layers):
    """
    Takes a normalizer adapted to input and the form of neural network and return keras model
    :param normalizer: normalizer
    :param layers: int list [m,n,..] meaning a first hidden layer with m neurones,
    a second hidden layer with n neurones, etc.
    :return: keras model
    """
    model = tf.keras.Sequential([normalizer])
    for i in range(len(layers)):
        neurons_no = layers[i]
        if (i==0):
            model.add(tf.keras.layers.Dense(neurons_no, input_dim=2, activation='relu' ))
        else:
            model.add(tf.keras.layers.Dense(neurons_no, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def learning_curve(history):
    """
    Takes the losses during different epochs, and number of epochs to plot learning curve
    :param losses: History.history
    :param epoch_no: int
    :return:
    """
    losses = history['loss']
    epochs = np.arange(1,len(losses)+1,1)
    plt.plot(epochs, losses)

def retransform(dataset, predicted_labels):
    """
    Takes the featers and predicted targets and makes one dataframe
    """
    retransformed_dataset = dataset.join(pd.DataFrame(predicted_labels, columns = ['Barrier']))
    return retransformed_dataset

def plot_surface_diff(data1,data2, key1, key2, key3, ax = None):
    diff = data1[key3] - data2[key3]
    data_diff = retransform(data1[[key1,key2]],diff)
    plot_surface(data_diff, key1, key2, key3, ax)
    return 0

if __name__ == "__main__":
    # Get the data
    data =pash_to_dataframe("barrier/pash_step3new.dat")
    train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(data, ["epsilon", "a3"], "Barrier")
    normalizer = normalize(train_features)
    model = build_model(normalizer, [500])
    model = learning_curve(model, train_features, train_labels, 2000)
    # Fit model
    plt.plot (test_labels, model.predict(test_features), '.')
    plt.show()
