import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import re
from launch_barrier import launch_barrier, input_template

def pash_to_type_1_barrier(path, start_indice=0):
    '''
    Takes a pash.dat file given out by type 1 barrier to return the value of barrier
    :param path: path to file
    :param start_indice:
    :return: float
    '''
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    line = lines[9]
    i = 0
    while i < (len(line)):
        if line[i] == "-":
            line = line[:i] + " " + line[i:]
            i += 1
        i += 1
    # Turn the line into a scalars array
    line = line.strip(" \n")
    line = re.split(r"\s+", line)
    line = np.array(line).astype(float)
    barrier = line[6]
    return barrier


def pash_type_1_to_dataframe(epsilon, alpha_3, barrier, index, features, target):
    '''
    Takes a value of epsilon, alpha_3 and barrier and returns a dataframe with these three values
    :param epsilon: float
    :param alpha_3: float
    :param barrier: float
    :param barrier: float
    :param barrier: float
    :return: DataFrame
    '''
    columns = features + [target]
    data_point = pd.DataFrame([[epsilon, alpha_3, barrier]], index=[index], columns=columns)
    return data_point


def change_input_type_1(epsilon, alpha_3):
    """
    :param epsilon: float
    :param alpha_3: float
    :return:
    """
    f = open("barrier/barrier.inp", "r")
    lines = f.readlines()
    f.close()
    lines[3]=' epsil={:.4f},alpha1=0.00,alpha2=0.00,alpha3={:.4f},alpha4=-.16D0, /END\n'.format(epsilon,alpha_3)
    f = open("barrier/barrier.inp", "w")
    f.write("".join(lines))
    f.close()

def launch_barrier_type_1(epsilon, alpha_3, index):
    '''
    Takes a value for epsilon, alpha_3 and an index at which this data should be and returns a DataFrame
    :param epsilon: float
    :param alpha_3: float
    :param index: int
    :return: DataFrame
    '''
    input_template("type1")
    change_input_type_1(epsilon, alpha_3)
    launch_barrier()
    barrier = pash_to_type_1_barrier("barrier/pash.dat")
    data_point = pash_type_1_to_dataframe(epsilon, alpha_3, barrier, index, ["epsilon", "a3"], "Barrier")
    return data_point