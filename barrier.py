import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import extract_pash
import launch_barrier
from scipy.interpolate import griddata

launch_barrier.input_template("step3")
launch_barrier.launch_barrier()
launch_barrier.change_file_name("pash.dat", "pash_step3.dat")
pash_data = extract_pash.pash_to_dataframe("barrier/pash_step3.dat")

x, y, z = extract_pash.plot_surface(pash_data, "P(1)", "P(2)", "Barrier")
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False)

plt.show()



