"""
Functions to change the input and run the barrier code and store its data output
"""
import subprocess
import os
import shutil
import numpy as np
import pandas as pd
import re

def input_template(template_name):
    """
    Changes barrier.inp to a template from barrier/template.
    Available templates: RUN1, RUN2, RUN3, RUN4, step3, test, type1, type2
    :param template_name: (str) name, no path, no .dat
    """
    source = "barrier/input_templates/"+template_name+".inp"
    shutil.copy(source, "barrier/barrier.inp")
    return 0

def change_input(epsilon, alpha3):
    """
    Takes values for epsilon and alpha_3 in the form of lists with min value, max value
    and the last value is either the number of points needed in between min and max values or the incrementation step.
    Changes the barrier.inp (of type 2 template) with these input values for epsilon and alpha_3.
    :param epsilon: (list) [min, max, n]
    :param alpha3: (list) [min, max, n]
    :return: None
    """
    if type(epsilon[2]) is int:
        epsilon_step = (epsilon[1]-epsilon[0])/epsilon[2]
    else:
        epsilon_step = epsilon[2]
    if type(alpha3[2]) is int:
        alpha3_step = (alpha3[1]-alpha3[0])/alpha3[2]
    else:
        alpha3_step = alpha3[2]
    f = open("barrier/barrier.inp", "r")
    lines = f.readlines()
    f.close()
    lines[3]='   vareps=.t.,beg1v={:.4f},step1v={:.4f},end1v={:.4f},\n'.format(epsilon[0],epsilon_step, epsilon[1])
    lines[4]='   varp3=.t., beg2v={:.4f},step2v={:.4f},end2v={:.4f} / END\n'.format(alpha3[0],alpha3_step, alpha3[1])
    f = open("barrier/barrier.inp", "w")
    f.write("".join(lines))
    f.close()

def launch_barrier ():
    """
    Launches the barrier.exe code.
    """
    os.chdir("barrier")
    subprocess.call("barrier.exe")
    os.chdir("..")
    return 0

def change_file_name (name, new_name, pash_to_data = True, keep_original = True):
    """
    Changes file name into new_name.
    Inside the barrier directory and while keeping the original file by default.
    :param name: (str) old file name
    :param new_name: (str) new file name
    :param pash_to_data: (bool) if True, will operate from barrier/ to data/pash
    :param keep_original: (bool) if True, will not delete the original file
    :return: None
    """
    if pash_to_data:
        name = "barrier/"+name
        new_name = "data/pash/"+new_name
    if keep_original:
        shutil.copy(name, new_name)
    else:
        os.rename(name, new_name)


def pash_to_dataframe(path, new_P1="epsilon", new_P2="a3", start_indice = 0):
    """
    Takes the path to a pash.dat file from a type 2 run and returns a pandas DataFrame.
    The keys P(1),P(2) are respectively renamed to new_P1, new_P2.
    :param path: (str) path to a file
    :param new_P1: (str) new name for the variable P(1)
    :param new_P2: (str) new name for the variable P(2)
    :param start_indice: (int) first index in the created DataFrame
    :return data: (pandas DataFrame) DataFrame containing all the variables and data from the input file
    """
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    columns = lines[9].strip(" \n")
    columns = re.split(r"\s+", columns)
    columns = np.array(columns)
    columns[columns == "P(1)"] = new_P1
    columns[columns == "P(2)"] = new_P2
    # Create an empty DataFrame to be filled
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
        # Turn the line into a scalars array and append it to the DataFrame
        line = line.strip(" \n")
        line = re.split(r"\s+", line)
        line = np.array(line).astype(float)
        line = pd.DataFrame([line],index=[index], columns=columns)
        data = data.append(line)
        index += 1

    return data


def pash_to_type_1_barrier(path):
    '''
    Takes the path to a pash.dat file from a type 1 run and returns the value of fission barrier.
    :param path: (str) path to file
    :return barrier: (float) fission barrier value
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
    # Take only the value for "Barrier"
    barrier = line[6]
    return barrier


def pash_type_1_to_dataframe(epsilon, alpha_3, barrier, index, features, target):
    '''
    Takes a value of epsilon, alpha_3 and fission barrier and returns a DataFrame of these three values
    with the line given by index and columns given by features key and target key.
    :param epsilon: (float) epsilon value
    :param alpha_3: (float) alpha_3 value
    :param barrier: (float) fission barrier value
    :param barrier: (int) index of line in the created DataFrame
    :param features: (str list) list of keys for features
    :param target: (str) key for target
    :return data_point: (pandas DataFrame) DataFrame containing the line with epsilon, alpha_3 and fission barrier
    '''
    columns = features + [target]
    data_point = pd.DataFrame([[epsilon, alpha_3, barrier]], index=[index], columns=columns)
    return data_point


def change_input_type_1(epsilon, alpha_3):
    """
    Takes values for epsilon and alpha_3.
    Changes the barrier.inp (of type 1 template) with these input values for epsilon and alpha_3.
    :param epsilon: (float) epsilon value
    :param alpha_3: (float) alpha_3 value
    :return: None
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
    Takes a value for epsilon, alpha_3 to run barrier.exe (type 1 run) returns a DataFrame with the line given by index.
    :param epsilon: (float) epsilon value
    :param alpha_3: (float) alpha_3
    :param index: (int) index of line in the created DataFrame
    :return data_point: (pandas DataFrame) DataFrame containing the line with epsilon, alpha_3 and fission barrier
    '''
    input_template("type1")
    change_input_type_1(epsilon, alpha_3)
    launch_barrier()
    barrier = pash_to_type_1_barrier("barrier/pash.dat")
    data_point = pash_type_1_to_dataframe(epsilon, alpha_3, barrier, index, ["epsilon", "a3"], "Barrier")
    return data_point


if __name__ == "__main__":
    input_template("step3")
    change_input(epsilon=[-0.1,0.4,0.1], alpha3=[0,0.2,0.1])
    launch_barrier()
    change_file_name("pash.dat", "pash_step3.dat", pash_to_data=True)
    data = pash_to_dataframe("data/pash/pash_step3.dat", new_P1='epsilon', new_P2='a3')
    print(data)
    print ("Finished")


