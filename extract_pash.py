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
    :return:
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



if __name__ == "__main__":
    print (pash_to_dataframe()["P(2)"])
    print ("runned")

