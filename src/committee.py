"""
Functions regarding the class Committee, managing a committee of models
"""
from nn_regression import create_datasets, normalize, build_model, retransform
import numpy as np
from analysis import pash_to_dataframe, plot_contour, plot_points
import matplotlib.pyplot as plt
from hyperparameters import calculate_mse
import os
import shutil
import tensorflow as tf

class Committee:
    """
    A committee of identical models, initialized with the number of models,
    The models are automatically built, and fit when the appropriate functions are called
    """
    def __init__(self, models_number):
        """
        Initialzations
        :param models_number: number of models in the committee
        """
        self.models_number = models_number
        self.models = []
    def build_model(self, normalizer, layers, optimizer="adam"):
        """
        Build and compile the models of the committee
        :param normalizer: the first normalizer layer of the models
        :param layers: list of ints, [# neurons in layer1, ... # neurons in layern]
                        containing the shape of the networks
        :param optimizer: Optimizer of the backpropagation
        :return: nothing
        """
        for i in range (self.models_number):
            self.models += [build_model(normalizer, layers, optimizer=optimizer)]

    def fit(self, train_features, train_labels, epochs,batch_size=30, verbose = 1, split_train = False, bootstrap = None):
        """
        Fit the models with the train_features over some epochs epochs
        :param train_features: DataFrame, [DataFrame1, DataFrame2....]
                                If a list (of len n_model) is given, each element is given
                                    to the respective model
                                Else has to be a dataframe
                                The input training data
        :param train_labels: As above
                                The expected labels from the training data
        :param epochs: int, Number of epochs of training
        :param batch_size: int, batch size
        :param verbose: Bool, if 1, prints the loss function at each epoch
                              if 0, only prints when each model starts the training
        :param split_train: Bool, if True, the training set is divided into n_models parts
                                    as possible
                            Overwritten by a float bootstrap
                            Overwritten by a list of train features
        :param bootstrap: None, float, if float, each model trains on a sample of the training set
                            Overwrites split_train
                            Overwritten by a list of train features
        :return: history_list: list of the output of each fit
        """
        history_list = []
        if split_train is True:
            n = len(train_features)//self.models_number
        i = 0
        for j in range(len(self.models)):
            model = self.models[j]
            print("train model "+str(i+1)+"/"+str(self.models_number))
            if type(train_features) is list:
                # Case where the training set is already split
                train_features_spec = train_features[j]
                train_labels_spec = train_labels[j]
            elif type(bootstrap) is float:
                # Bootstrap method: each model takes a sample
                train_features_spec = train_features.sample(frac=bootstrap)
                indexes = train_features_spec.index
                train_labels_spec = train_labels[indexes]
            elif split_train:
                # Case where we have to split the training set in equal parts
                train_features_spec = train_features.sample(frac=1./(self.models_number-i))
                indexes = train_features_spec.index
                train_labels_spec = train_labels[indexes]
                train_features = train_features.drop(indexes)
                train_labels = train_labels.drop(indexes)
            else:
                # Each model trains on the same training set
                train_features_spec = train_features
                train_labels_spec = train_labels
            history_list += [model.fit(train_features_spec, train_labels_spec, batch_size=batch_size, epochs=epochs, verbose=verbose)]
            i += 1
        return history_list
    def predict (self, data_features):
        """
        Predict the output from the input features
        :param data_features: dataframe containing the input
        :return: list of arrays containing the predictions from each model
        """
        list_prediction = []
        for model in self.models:
            list_prediction += [model.predict(data_features)]
        return list_prediction

def get_mean_var(list_prediction):
    """
    Get the mean prediction and the variance over it from the output of committee.predict
    :param list_prediction: list of arrays, output of committee.predict
    :return: mean_list: list of the mean prediction from all models
    :return: var_list: list of the standard on the prediction from all models
    """
    mean_list = np.mean(list_prediction, 0)
    var_list = np.var(list_prediction, 0)
    return mean_list, var_list

def plot_histo (committee, point_features, expected_value, ax=plt.gca()):
    """
    plot the histogram of the prediction on one point by each model of a committee
    Adds a vertical bar for the mean and for the expected value
    :param committee: Committee object
    :param point_features: single entry dataframe
    :param expected_value: expected value on the point
    :param ax: ax on which to plot the histogram
    :return:
    """
    predicted_list = np.ravel(committee.predict(point_features))
    n, b, p= ax.hist(predicted_list)
    ax.vlines(expected_value, 0, max(n), colors="r")
    ax.vlines(np.mean(predicted_list), 0, max(n), colors="g")
    ax.set_xlabel('Barrier')


class HistOnClick:
    """
    This class makes the histogram of the values for a point, predicted
    by different models of the committee when the point is clicked.
    It adds the mean predicted value and the expected value.
    """
    def __init__(self, ax1, ax2, committee, data, features, target, in_order = True):
        """
        :param ax1: ax where the points can be clicked
        :param ax2: ax on which the histogram is plotted
        :param committee: a Committee object
        :param data: data from which the expected target comes
        :param features: list of str, list of keys
        :param target: str, key of the target
        :param in_order: say True if the dataset is ordered in a grid pattern. It makes the function quicker
        """
        self.committee = committee
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_click)

        self.in_order = in_order
        self.data = data
        self.features = features
        self.target = target
        # If the data is in agrid pattern , determine the step, min and max in both dimensions
        if in_order is True:
            index = 0
            diff = 1
            while diff > 0:
                diff = data.loc[index+1, features[1]]-data.loc[index, features[1]]
                index += 1
            lent = len(data)
            self.min1 = data.loc[0, features[0]]
            max1 = data.loc[lent-1, features[0]]
            self.min2 = data.loc[0, features[1]]
            len2 = index
            len1 = lent // index
            self.len2 = len2
            max2 = data.loc[len2 - 1, features[1]]
            self.step1 = (max1 - self.min1 ) / len1
            self.step2 = (max2 - self.min2 ) / len2
    def on_click (self, event):
        """
        On a click, plot the histogram corresponding to the point
        :param event: from the click event
        """
        features = self.features
        target = self.target
        data = self.data
        # Get the point coordinates
        coord = np.array([event.xdata, event.ydata])
        # If not in order, find the closest point in data with  brute force
        if self.in_order is False:
            index = 0
            norm = 1000
            for i in range(len(data)):
                point_features = data.loc[[i], features]
                newnorm = np.linalg.norm(point_features-coord)
                if newnorm < norm :
                    norm = newnorm
                    index = i
        # if it is ordered, get its position in data
        else:
            pos1 = round((coord[0] - self.min1 )/self.step1 - 1)
            pos2 = round((coord[1] - self.min2 )/self.step2 - 1)
            index = int(pos1*self.len2 + pos2)
        self.ax2.cla()
        self.ax1.set_data(data.loc[[index], features[0]], data.loc[[index], features[1]])
        point_features = data.loc[[index], features]
        expected_value = data.loc[[index], target]
        plot_histo(self.committee, point_features, expected_value, self.ax2)
        self.ax2.figure.canvas.draw()

def interactive_plots(committee, features, target, data, train_dataset, plot_train=True):
    """
    Plots many interesting plots for a committee: Variance, histogram, predicted and absolute error
    :param committee: Committee object
    :param features: list of str, keys
    :param target: str, key
    :param data: dataframe to test
    :param train_dataset: dataframe, train dataset will be plotted if plot_train is True
    :param plot_train: bool, if True, plot the training  dataset
    :return:
    """
    list_prediction = committee.predict(data[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_target = retransform(data[features], predicted_target)
    variance = retransform(data[features], variance)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    plot_contour(variance, features[0], features[1], target, colorbar=True, levels=40, ax=ax1, bar_name="Variance")
    plot_contour(predicted_target, features[0], features[1], target, colorbar=True, levels=40, ax=ax3,
                 bar_name="Predicted Barrier")

    diff = np.abs(data[target] - predicted_target[target])
    data_diff = retransform(data[features], diff)
    plot_contour(data_diff, features[0], features[1], target, colorbar=True, ax=ax4, levels=40,
                 bar_name="Barrier difference with expected")

    if plot_train:
        plot_points(train_dataset, features, ax1)
        plot_points(train_dataset, features, ax3)
        plot_points(train_dataset, features, ax4)

    pointer, = ax1.plot([0], [0], "+")
    object = HistOnClick(pointer, ax2, committee, data, features, target)

    plt.show()
    return object

def save_committee(dir_name, committee):
    """
    Save a committee
    :param dir_name: str, name of the committee
    :param committee: Committee object
    :return: None
    """
    os.chdir("data/models/")
    if dir_name in os.listdir():
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    os.chdir(dir_name)
    i = 0
    for model in committee.models:
        model.save("model"+str(i))
        i += 1
    os.chdir("../../..")

def load_committee(dir_name):
    """
    Load a committee from a file
    :param dir_name: str, name of the committee
    :return: Committee object
    """
    models = []
    n_models = 0
    os.chdir("data/models/"+dir_name)
    for filename in os.listdir():
        models += [tf.keras.models.load_model(filename)]
        n_models += 1
    committee = Committee(n_models)
    committee.models = models
    os.chdir("../../..")
    return committee

if __name__ == "__main__":
    # Get the data
    features = ["epsilon", "a3"]
    data = pash_to_dataframe("data/pash/large_pash.dat")
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(data, features, "Barrier", frac=0.1)
    print("Loaded data")
    # Create a committee
    """
    committee = Committee(50)
    normalizer = normalize(train_features)
    committee.build_model(normalizer, [150, 150, 150], optimizer="adamax")
    # Fit on the data
    committee.fit(train_features, train_labels, epochs=2000, verbose=0, bootstrap=.1)
    # Example on saving and loading a committee
    print("Saving model")
    save_committee("testing", committee)
    """
    print("Loading model")
    committee = load_committee("testing")
    # Compute the MSE
    list_prediction = committee.predict(data[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_target = np.ravel(predicted_target)
    rmse = np.sqrt(calculate_mse(predicted_target, data["Barrier"]))
    print("rmse ", rmse, "MeV")
    # Plot the variance, the prediction, difference with expected values and the histograms
    o = interactive_plots(committee, features, "Barrier", data, train_dataset, plot_train=False)

