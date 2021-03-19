"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import re

def pash_to_dataframe(lines):
    """
    Takes the lines from a pash.dat file and returns a pandas Dataframe
    """

    columns = lines[9].strip(" \n").split("  ")
    data = pd.DataFrame(columns = columns)

    for line in lines[10:]:
        line = line.strip(" \n")
        line = re.split(r"\s{1,}",line)
        line = np.array(line).astype(float)
        line = pd.DataFrame([line],columns = columns)
        data = data.append(line)

    return data

def plot_surface(data, key_1, key_2, key_3):
    """
        Takes the DataFrame file with the keys of interest to give x, y and z to be plotted
    """
    x = np.linspace(data[key_1].min(), data[key_1].max(), len(data[key_1].unique()))
    y = np.linspace(data[key_2].min(), data[key_2].max(), len(data[key_2].unique()))
    x1, y1 = np.meshgrid(x, y)
    z1 = griddata((data[key_1], data[key_2]), data[key_3], (x1, y1), method='cubic')

    return x1, y1, z1

if __name__ == "__main__":
    data = pash_to_dataframe()
    print (data)
    #plot_surface(data, "P(1)", "P(2)", " Barrier")
    print ("runned")

