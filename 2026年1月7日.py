# %% Yセンサ値生成プログラム
import numpy as np
import os 

save_folder = input("input save folder path: ")
pp_value = input("input pp value: ")
pixel_size = int(input("input pixel size"))**2
pp_sensor = float(pp_value)/30 # Vに変換

txt_path_1024 = r"C:\nojima\AFM6measurement\260224\アライメントチェック\1402_3\Ysensors.txt"
original_data_1024 = np.loadtxt(txt_path_1024)
data_num = 1024*1024

data = []
if pixel_size == 512*512:
    for i in range(data_num):
        if i % int(data_num/pixel_size) == 0:
            data.append(original_data_1024[i])
elif pixel_size == 1024 * 1024:
    data = original_data_1024

print(np.max(data))
normalized_data = data / (np.max(data) - np.min(data))
print(np.max(normalized_data), np.min(normalized_data))

newdata = normalized_data * pp_sensor
print(np.max(newdata), np.min(newdata))

np.savetxt(os.path.join(save_folder, "Ysensors.txt"), newdata)

#%% 
import numpy as np
import os 
import matplotlib.pyplot as plt

folder1 = r"C:\nojima\AFM6measurement\260319\1932_3"
folder2 = r"C:\nojima\AFM6measurement\260319\1932_ゲイン5000"

youngs = []
xsensor = []
ysensor = []
for folder in [folder1, folder2]:
    data = np.load(os.path.join(folder, "AFM_Analysis_Results\youngs_modulus_map.npz"))
    youngs.append(np.log10(data['map_data']))

# 同一スケールでimshow
vmin = min(np.min(youngs[0]), np.min(youngs[1]))
vmax = max(np.max(youngs[0]), np.max(youngs[1]))
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(youngs[0], cmap='afmhot', vmin=vmin, vmax=vmax)
axes[0].set_title('Folder 1 Young\'s Modulus Map (log10)')
axes[1].imshow(youngs[1], cmap='afmhot', vmin=vmin, vmax=vmax)
axes[1].set_title('Folder 2 Young\'s Modulus Map (log10)')
plt.show()

median_1 = []
median_2 = []
rows, cols = youngs[0].shape
for i in range(cols):
    median_1.append(np.median(youngs[0][:,i]))
    median_2.append(np.median(youngs[1][:,i]))


plt.plot(median_1, label="Folder 1")
plt.plot(median_2, label="Folder 2")
plt.legend()
plt.title("Median of log10(Young's Modulus) for each column")
plt.xlabel("Column Index")
plt.ylabel("Median of log10(Young's Modulus)")
plt.show()
# %%
# マップ全体のヒストグラムをプロット
plt.hist(youngs[0].flatten(), bins=100, alpha=0.5, label='Folder 1')
plt.hist(youngs[1].flatten(), bins=100, alpha=0.5, label='Folder 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of log10(Young\'s Modulus)')
plt.legend()
plt.show()
# %%
# %%
row_start = 500
row_end = 1000
youngs_cropped = []
for i in range(cols):
    youngs_cropped.append(np.median(youngs[0][row_start:row_end, i]))
plt.plot(youngs_cropped)
# %%
import numpy as np
import matplotlib.pyplot as plt

filepath = r"C:\nojima\AFM6measurement\260319\1932_スイープ計測\AFM_Analysis_Results\youngs_modulus_map.npz"
data = np.load(filepath)
map_data = np.log10(data['map_data'])
print(data.files)
# %%
row = 1000
plt.plot(map_data[row, :])
plt.title(f"Row {row} of log10(Young's Modulus) Map")
plt.xlabel("Column Index")
plt.ylabel("log10(Young's Modulus)")
plt.show()
# %%
row, col = np.shape(map_data)
std = []
for i in range(row):
    std.append(np.std(map_data[i, :]))
plt.plot(std)
plt.title("Standard Deviation of Each Row in log10(Young's Modulus) Map")
plt.xlabel("Row Index")
plt.ylabel("Standard Deviation")
plt.show()
# %%