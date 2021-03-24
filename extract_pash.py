"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
import pandas as pd
import re
from launch_barrier import launch_barrier, input_template

def pash_to_dataframe(path, new_P1="epsilon", new_P2="a3", start_indice = 0):
    """
    Takes the path to a pash.dat file and returns a pandas Dataframe
    compatible to run type 2
    The keys P(1),P(2) are respectively renamed to new_P1, new_P2
    :param path: str, path to a file
    :param new_P1: new name for the variable P(1)
    :param new_P2: new name for the variable P(2)
    :return: a pandas dataframe containing all the variables and data from the input file
    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    columns = lines[9].strip(" \n")
    columns = re.split(r"\s+", columns)
    columns = np.array(columns)
    columns[columns == "P(1)"] = new_P1
    columns[columns == "P(2)"] = new_P2
    # Create an mpty dataframe to be filled
    data = pd.DataFrame(columns=columns)
    index = start_indice
    for line in lines[10:]:
        # The following condition adds a space before - signs to avoid number concatenation
        i=0
        while i < (len(line)):
            if line[i] == "-":
                line = line[:i]+" "+line[i:]
                i+=1
            i+=1
        # Turn the line into a scalars array and append it to the dataframe
        line = line.strip(" \n")
        line = re.split(r"\s+", line)
        line = np.array(line).astype(float)
        line = pd.DataFrame([line],index=[index], columns=columns)
        data = data.append(line)
        index += 1

    return data

def plot_surface(data, key_1, key_2, key_3, ax = None, alpha = 1):
    """
    Takes the DataFrame with the keys of interest to be plotted on a 3D surface
    The input data should cover the whole surface (i.e. avoid holes)
    :param data: a DataFrame
    :param key_1: x axis' key of the DataFrame
    :param key_2: y axis' key of the DataFrame
    :param key_3: z axis' key of the DataFrame
    :param ax: if given, figure will be plotted on it
    :param alpha: transparency of the surface
    :return: Nothing
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

def plot_heatmap(data, key_1, key_2, key_3, ax = None, colorbar = True, cmap="hot"):
    """
    Takes the DataFrame with the keys of interest to be plotted on a 2D heatmap
    The input data should cover the whole surface (i.e. avoid holes)
    :param data: a DataFrame
    :param key_1: x axis' key of the DataFrame
    :param key_2: y axis' key of the DataFrame
    :param key_3: z axis' key of the DataFrame
    :param ax: if given, figure will be plotted on it
    :param colorbar: if True, will add a colorbar
    :param cmap: Color map of the figure
    :return: Nothing
    """
    x = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x, y)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    if ax is None:
        plt.imshow(z, cmap=cmap)
        if colorbar:
            plt.colorbar()
    else:
        ax.imshow(z,cmap=cmap)
        if colorbar:
            ax.colorbar()

def plot_contour(data, key_1, key_2, key_3, levels=6, ax=plt.gca(), cmap="hot", colorbar=True, bar_name=None):
    """
    Plot a contour graph of the input data
    :param data: a DataFrame
    :param key_1: x axis' key of the DataFrame
    :param key_2: y axis' key of the DataFrame
    :param key_3: z axis' key of the DataFrame
    :param ax: if given, figure will be plotted on it
    :return:
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

    ax.plot(data[features[0]], data[features[1]], '.')

if __name__ == "__main__":
    #input_template("step3")
    #launch_barrier()
    data = pash_to_dataframe("barrier/large_pash.dat")
    print (data)
    plot_contour(data, "epsilon", "a3", "Barrier", levels=100, cmap="viridis")
    plt.show()
    print("ran")


