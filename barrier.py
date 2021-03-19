import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import extract_pash

f = open("barrier/pash.dat", "r")
lines = f.readlines()
f.close()
pash_data = extract_pash.pash_to_dataframe(lines)
print(pash_data)


