"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import re

def pash_to_dataframe(path):
    """
    Takes the lines from a pash.dat file and returns a pandas Dataframe
    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    columns = lines[9].strip(" \n")
    columns = re.split(r"\s+", columns)
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

def plot_surface(data, key_1, key_2, key_3):
    """
        Takes the DataFrame file with the keys of interest to give x, y and z to be plotted
    """
    x1 = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y1 = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x, y = np.meshgrid(x1, y1)
    z = griddata((data[key_1], data[key_2]), data[key_3], (x, y), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False)


if __name__ == "__main__":
    data = pash_to_dataframe("barrier/pash_step3.dat")
    print (data.columns)
    plot_surface(data, "P(1)", "P(2)", "Barrier")
    print ("ran")

