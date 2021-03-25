import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from launch_barrier import pash_to_dataframe
import tensorflow as tf
from nn_regression import retransform


def plot_surface(data, key_1, key_2, key_3, ax = None, alpha = 1):
    """
    Takes the DataFrame with the keys of interest to be plotted on a 3D surface.
    The input data should cover the whole surface (i.e. avoid holes).
    :param data: (pandas DataFrame) initial DataFrame
    :param key_1: (str) x axis' key of the DataFrame
    :param key_2: (str) y axis' key of the DataFrame
    :param key_3: (str) z axis' key of the DataFrame
    :param ax: (axes) if given, figure will be plotted on it
    :param alpha: (float) transparency of the surface
    :return: None
    """
    x1 = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y1 = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x1, y1)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=alpha)
    ax.set_xlabel(key_1)
    ax.set_ylabel(key_2)
    ax.set_zlabel(key_3)

def plot_surface_diff(data1, data2, key1, key2, key3, ax = None):
    """
    Takes two DataFrames with the keys of interest and plots the difference between key3 of DataFrames 1 & 2.
    :param data1: (pandas DataFrame) DataFrame 1
    :param data2: (pandas DataFrame) DataFrame 2
    :param key1: (str) x axis' key of the DataFrame
    :param key2: (str) y axis' key of the DataFrame
    :param key3: (str) z axis' key of the DataFrame, on which the difference is made
    :param ax: on which the surface is plotted
    :return: None
    """
    diff = data1[key3] - data2[key3]
    data_diff = retransform(data1[[key1, key2]], diff)
    plot_surface(data_diff, key1, key2, key3, ax)


def plot_contour(data, key_1, key_2, key_3, levels=6, ax=plt.gca(), cmap="hot", colorbar=True, bar_name=None):
    """
    Takes the DataFrame with the keys of interest and plots a contour graph of the input data
    :param data: (pandas DataFrame) DataFrame 1
    :param key_1: (str) x axis' key of the DataFrame
    :param key_2: (str) y axis' key of the DataFrame
    :param key_3: (str) z axis' key of the DataFrame
    :param levels: (int) number contour levels - 6 by default
    :param ax: (axes) if given, figure will be plotted on it
    :param cmap: (str) choice of colormap - 'hot' by default
    :param colorbar: (bool) if True displays colorbar
    :param bar_name: (str) name to be displayed next to colorbar
    :return: None
    """
    x = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x, y)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')

    img = ax.contourf(x,y,z, levels=levels, cmap=cmap)
    ax.set_xlabel(key_1)
    ax.set_ylabel(key_2)
    if colorbar:
        if bar_name is None:
            bar_name = key_3
        plt.colorbar(img, label=bar_name, ax=ax)

def plot_points(data, features, ax=plt.gca()):
    """
    Takes the DataFrame with the keys of features used to place the points of training datasets
    :param data: (pandas DataFrame) DataFrame 1
    :param features: (str list) list of keys of features
    :param ax: (axes) if given, figure will be plotted on it
    :return: None
    """
    ax.plot(data[features[0]], data[features[1]], '.')

def calculate_mse(predicted_values, real_values):
    """
    Takes DataFrames to calculate the mean squared error between them.
    :param predicted_values: (pandas DataFrame) estimated values
    :param real_values: (pandas DataFrame) expected values
    :return mse: (float) mean squared error
    """
    no_points = len(predicted_values)
    error = predicted_values - real_values
    mse = np.sum(error * error) / no_points
    return mse

if __name__ == "__main__":
    features = ["epsilon", "a3"]
    # Import data
    data = pash_to_dataframe("data/pash/large_pash.dat")
    # Plot a 3D surface
    plot_surface(data, "epsilon", "a3", "Barrier")
    plt.show()
    # Import a model and predict the data
    model = tf.keras.models.load_model("data/models/example_NN")
    predicted_values = model.predict(data[features])
    predicted_values = retransform(data[features], predicted_values)
    # Plot the difference between predicted and expected values
    plot_surface_diff(data, predicted_values, "epsilon", "a3", "Barrier")
    plt.show()
    # Plot as a 2D heatmap
    fig, (ax1, ax2) = plt.subplots(1,2)
    plot_contour(predicted_values,"epsilon", "a3", "Barrier", bar_name="Predicted Barrier", ax=ax1, levels=30)
    plot_contour(data,"epsilon", "a3", "Barrier", bar_name="Expected Barrier", ax=ax2, levels=30)
    plt.show()
    # Compute the RMSE
    RMSE = np.sqrt(calculate_mse(predicted_values['Barrier'], data['Barrier']))
    print("The RMSE is ", RMSE, "MeV")
    print("Finised")