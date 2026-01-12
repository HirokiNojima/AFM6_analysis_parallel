#%%
import numpy as np
import nptdms
import os

data = np.load(r"C:\nojima\AFM6measurement\260107\1901_sensor_gain200\AFM_Analysis_Results\youngs_modulus_map.npz")
# %%
print(data.files)
print(data['map_data'].shape)
# %%
row, col = data['map_data'].shape
col_average = []
for c in range(col):
    col_average.append(np.nanmean(data['map_data'][:100, c]))

import matplotlib.pyplot as plt
plt.plot(col_average)
plt.show()
# %%
# %%
import matplotlib.pyplot as plt
plt.plot(data['map_data'][300, :])
plt.show()
# %%
