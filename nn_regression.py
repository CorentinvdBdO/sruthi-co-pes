import tensorflow as tf
from extract_pash import pash_to_dataframe
from launch_barrier import launch_barrier, input_template


if __name__ == "__main__":
    data = pash_to_dataframe("barrier/pash.dat")
    print(data.drop(columns = ['Sh.Cor', 'Eldm', 'dU(N)', 'dU(P)', 'dP(N)','dP(P)', 'BC', 'BS']))
    print (data.columns)
    print ("ran")