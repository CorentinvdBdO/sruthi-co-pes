"""
Test file for Corentin when running random bits of code
"""
import subprocess
import os

if __name__ == "__main__":
    os.chdir("barrier")
    subprocess.call("barrier.exe")
    # barrier/barrier.exe
    print ("runned")


