from launch_barrier import launch_barrier, input_template, change_file_name
from extract_pash import pash_to_type_1_barrier, pash_type_1_to_dataframe

input_template("type1")
launch_barrier()
change_file_name("pash.dat", "pash_type1_test.dat")
barrier = pash_to_type_1_barrier("barrier/pash_type1_test.dat")
data_point = pash_type_1_to_dataframe(0.5, 0.1, barrier, 0, ["epsilon", "alpha_3"], "Barrier")
print(data_point)
print(barrier)
