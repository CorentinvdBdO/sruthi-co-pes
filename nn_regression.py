import tensorflow as tf
from extract_pash import pash_to_dataframe
from launch_barrier import launch_barrier, input_template


if __name__ == "__main__":
    f = open("barrier/pash.dat", "r")
    lines = f.readlines()
    f.close()
    data = pash_to_dataframe(lines)
    print(data)
    print (data.columns)
    print ("ran")