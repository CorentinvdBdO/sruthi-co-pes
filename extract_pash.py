"""
takes the pash.dat file and turns it into manageable data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def pash_to_dataframe():
    """
    Reads the file pash.dat and returns a pandas Dataframe
    """
    f = open("barrier/pash.dat", "r")
    lines = f.readlines()
    f.close()

    columns = lines[9].strip(" \n").split("  ")
    data = pd.DataFrame(columns = columns)

    for line in lines[10:]:
        line = line.strip(" \n")
        line = re.split(r"\s{1,}",line)
        line = np.array(line).astype(float)
        line = pd.DataFrame([line],columns = columns)
        data = data.append(line)

    return data

def plot_surface (data, x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(data[x], data[y], data[z])
    plt.show

if __name__ == "__main__":
    data = pash_to_dataframe()
    print (data)
    plot_surface(data, "P(1)", "P(2)", "Barrier")
    print ("runned")

