import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import extract_pash

f = open("barrier/pash.dat", "r")
lines = f.readlines()
f.close()
pash_data = extract_pash.pash_to_dataframe(lines)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(pash_data["P(1)"], pash_data["P(2)"], pash_data["Barrier"])
plt.show()


