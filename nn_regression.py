import tensorflow as tf
import numpy as np
from extract_pash import pash_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_datasets(path, feature_key_1, feature_key_2, target_key, frac=0.5):
    """
    Takes a file and keys of interest to return two datasets - one for training and one for testing
    :param path: path to file with the output data of BARRIER
    :param feature_key_1: str key of first parameter x
    :param feature_key_2: str key of second parameter y
    :param target_key: str key of quantity of interest z = f(x,y)
    :return: dataset consisting of columns given by x, y and z seperated randomly into a training dataset
    and a test dataset
    """
    data = pash_to_dataframe(path)
    features = [feature_key_1, feature_key_2]
    target = target_key
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

def learning_curve(model, train_features, train_labels, epoch_no):
    """
    Takes a model, training features and labels and number of epochs to train the network
    :param model: model
    :param features: dataset
    :param labels: labels
    :param epoch_no: int
    :return:
    """
    losses = model.fit(train_features, train_labels, epochs=epoch_no).history
    epochs = np.arange(1,epoch_no,1)
    return model
    #plt.plot(epochs, losses)

def retransform(model, test_features, test_labels):
    """
    Takes trained model to predict values for test features
    :param model: model
    :param features: dataset
    :param labels: dataset
    :return: x, y, z to plot
    """
    predicted_labels = model.predict(test_features)
    y = test_features.pop('P(2)')
    x = test_features






if __name__ == "__main__":
    # Get the data
    train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets("barrier/pash.dat", "P(1)", "P(2)", "Barrier")
    normalizer = normalize(train_features)
    model = build_model(normalizer, [500])
    model = learning_curve(model, train_features, train_labels, 2000)
    # Build model
    # model = tf.keras.Sequential([normalizer,
    #     tf.keras.layers.Dense(500, input_dim=2, activation='relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    # Compile model

    # Fit model
    plt.plot (test_labels, model.predict(test_features), '.')
    plt.show()
