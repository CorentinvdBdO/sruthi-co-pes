"""
Test file for Corentin when running random bits of code
"""
import subprocess
import os
import shutil

def input_template(template_name):
    """
    Change barrier.inp to a template from barrier/template
    available templates: RUN1, RUN2, RUN3, RUN4, step3, test, type1, type2
    :param template_name: str name, no path, no .dat
    """
    source = "barrier/input_templates/"+template_name+".inp"
    shutil.copy(source, "barrier/barrier.inp")
    return 0


def launch_barrier ():
    """
    Launch the barrier.exe code
    """
    os.chdir("barrier")
    subprocess.call("barrier.exe")
    os.chdir("..")
    return 0

def change_file_name (name, new_name, in_barrier = True, keep_original = True):
    """
    Change file name into new_name
    Inside the barrier directory by default
    :param name: old file name
    :param new_name: new file name
    :param in_barrier: if True, will operate in barrier/
    :param keep_original: if True, will not delete the original file
    :return:
    """
    if in_barrier:
        os.chdir("barrier")
    if keep_original:
        shutil.copy(name, new_name)
    else:
        os.rename(name, new_name)
    if in_barrier:
        os.chdir("..")
    return 0




if __name__ == "__main__":
    #input_template("large")
    #launch_barrier()
    change_file_name("pash.dat", "large_pash.dat")
    print ("ran")


