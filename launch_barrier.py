"""
Test file for Corentin when running random bits of code
"""
import subprocess
import os
import shutil

def input_change ():
    """
    Change the parameters from barrier.inp
    """

def input_template(template_name):
    """
    Change barrier.inp to a template
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
    input_template("step3")
    launch_barrier()
    print ("ran")


