"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import re
from launch_barrier import launch_barrier, input_template

def pash_to_dataframe(path, new_P1="epsilon", new_P2="a3"):
    """
    Takes the path to a pash.dat file and returns a pandas Dataframe
    compatible to run type 2
    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    columns = lines[9].strip(" \n")
    columns = re.split(r"\s+", columns)
    columns = np.array(columns)
    columns[columns == "P(1)"] = new_P1
    columns[columns == "P(2)"] = new_P2
    dataframe = pd.DataFrame(columns=columns)
    index = 0
    for line in lines[10:]:
        # The following condition adds a space before - signs to avoid number concatenation
        i=0
        while i < (len(line)):
            if line[i] == "-":
                line = line[:i]+" "+line[i:]
                i+=1
            i+=1
        line = line.strip(" \n")
        line = re.split(r"\s+", line)
        line = np.array(line).astype(float)
        line = pd.DataFrame([line],index=[index], columns=columns)
        dataframe = dataframe.append(line)
        index += 1

    return dataframe

def plot_surface(data, key_1, key_2, key_3, ax = None, alpha = 1):
    """
        Takes the DataFrame with the keys of interest to give x, y and z to be plotted
    """
    x1 = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y1 = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x1, y1)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=alpha)
    else:
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=alpha)

def plot_heatmap(data, key_1, key_2, key_3, ax = None, colorbar = True, cmap="hot"):
    """
        Takes the DataFrame with the keys of interest to give x, y and z to be plotted
    """
    x1 = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y1 = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x1, y1)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    if ax is None:
        plt.imshow(z, cmap=cmap)
        if colorbar:
            plt.colorbar()
    else:
        ax.imshow(z,cmap=cmap)
        if colorbar:
            ax.colorbar()

def plot_contour(data, key_1, key_2, key_3, ax = None):
    x = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x, y)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    if ax is None:
        plt.contour(x,y,z)
    else :
        ax.contour(x,y,z)

if __name__ == "__main__":
    #input_template("step3")
    #launch_barrier()
    data = pash_to_dataframe("barrier/pash.dat")
    print (data)
    plot_contour(data, "epsilon", "a3", "Barrier")
    plt.show()
    print ("ran")


