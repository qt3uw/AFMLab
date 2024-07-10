import numpy as np
from matplotlib import pyplot as plt, colors
# from matplotlib import cm
# import skimage as ski

# convert piezo voltage data into an array
volt_data = np.loadtxt("image_data.txt", dtype=float)
print(volt_data)
print(np.size(volt_data))
# convert piezo voltage to height displacement
# *100 because .Normalize() normalizes between 0 and 1
height_data = colors.Normalize()(volt_data) * 100
print(height_data)
# create colormap of data
x, y = np.mgrid[0:20:0.2, 0:20:0.2]
V = volt_data
z = height_data
plt.pcolormesh(x, y, z, cmap='Greys')
plt.title("AFM Scan")
plt.xlabel('x pos [microns]')
plt.ylabel('y pos [microns]')
# plt.colorbar(label="Z piezo voltage [V]")
plt.colorbar(label="Z piezo displacement [nm]")
plt.show()

