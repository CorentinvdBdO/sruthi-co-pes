"""
Test file for Corentin when running random bits of code
"""
import subprocess
import os



def launch_barrier ():
    """
    Launch the barrier.exe code
    """
    os.chdir("barrier")
    subprocess.call("barrier.exe")
    os.chdir("..")
    return 0

def change_file_name (name, new_name, in_barrier = True):
    """
    Change file name into new_name
    """
    if in_barrier:
        os.chdir("barrier")
    os.rename(name,new_name)
    if in_barrier:
        os.chdir("..")
    return 0

if __name__ == "__main__":
    launch_barrier()
    print ("runned")


