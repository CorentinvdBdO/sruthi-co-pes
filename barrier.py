import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import extract_pash
import launch_barrier
from scipy.interpolate import griddata

launch_barrier.input_template("step3")
launch_barrier.launch_barrier()
launch_barrier.change_file_name("pash.dat", "pash_step3new.dat")
pash_data = extract_pash.pash_to_dataframe("barrier/pash_step3new.dat")
extract_pash.plot_surface(pash_data, "P(1)", "P(2)", "Barrier")

plt.show()



