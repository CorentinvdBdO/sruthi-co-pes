import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import extract_pash
from scipy.interpolate import griddata

f = open("barrier/pash.dat", "r")
lines = f.readlines()
f.close()
pash_data = extract_pash.pash_to_dataframe(lines)
print(pash_data[' Barrier'])

x, y, z = extract_pash.plot_surface(pash_data, "P(1)", "P(2)", " Barrier")
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False)

plt.show()



