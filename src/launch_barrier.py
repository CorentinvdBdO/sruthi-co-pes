"""
Functions used to deal with the barrier code
"""
import subprocess
import os
import shutil
import numpy as np
import pandas as pd
import re

def input_template(template_name):
    """
    Change barrier.inp to a template from barrier/template
    available templates: RUN1, RUN2, RUN3, RUN4, step3, test, type1, type2
    :param template_name: str name, no path, no .dat
    """
    source = "barrier/input_templates/"+template_name+".inp"
    shutil.copy(source, "barrier/barrier.inp")
    return 0

def change_input(epsilon, alpha3):
    """
    :param epsilon: list : [min, max, n]
    :param alpha3: list : [min, max, n]
    :return:
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
    Launch the barrier.exe code
    """
    os.chdir("barrier")
    subprocess.call("barrier.exe")
    os.chdir("..")
    return 0

def change_file_name (name, new_name, pash_to_data = True, keep_original = True):
    """
    Change file name into new_name
    Inside the barrier directory by default
    :param name: old file name
    :param new_name: new file name
    :param pash_to_data: if True, will operate from barrier/ to data/pash
    :param keep_original: if True, will not delete the original file
    :return:
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


if __name__ == "__main__":
    input_template("step3")
    change_input(epsilon=[-0.1,0.4,0.1], alpha3=[0,0.2,0.1])
    launch_barrier()
    change_file_name("pash.dat", "pash_step3.dat", pash_to_data=True)
    data = pash_to_dataframe("data/pash/pash_step3.dat", new_P1='epsilon', new_P2='a3')
    print(data)
    print ("Finished")


