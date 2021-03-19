"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
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

def plot_surface(data, key1, key2, key3):
    """
        Takes the DataFrame file with the keys of interest to give x, y and z to be plotted
    """
    x = np.linspace(data['key1'].min(), data['key1'].max(), len(data['key1'].unique()))
    y = np.linspace(data['key2'].min(), data['key2'].max(), len(data['key2'].unique()))
    x, y = np.meshgrid(x1, y1)
    z = griddata((data['key1'], data['key2']), data['key3'], (x, y), method='cubic')

    return x, y, z

if __name__ == "__main__":
    data = pash_to_dataframe()
    print (data)
    plot_surface(data, "P(1)", "P(2)", " Barrier")
    print ("runned")

