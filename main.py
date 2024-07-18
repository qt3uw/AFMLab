import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path
import os
import pandas as pd

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM Data')


def get_afm_data(folder_path):
    # Initialize an empty list to store data frames
    array_list = []

    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a numpy array
            with open(file_path, 'r') as f:
                df = pd.read_csv(f, delimiter=';', header=None)
                array = df.to_numpy()[:, :-1].astype(float)

            # Append the numpy array to the list
            array_list.append(array)

    return array_list


def create_image(filepath):
    # convert piezo voltage data into an array
    # filepath = Path(r"C:\Users\QT3\Documents\EDUAFM Data\TestSample_ConstantForce_StrainGauge_100px_100pps.csv")
    volt_data = np.loadtxt(filepath, dtype=float, delimiter=';')
    print(volt_data)
    print(np.size(volt_data))
    # convert piezo voltage to height displacement
    # *100 because .Normalize() normalizes between 0 and 1
    # height_data = colors.Normalize()(volt_data) * 100
    # print(height_data)
    # create colormap of data
    x, y = np.mgrid[0:20:0.2, 0:20:0.2]
    v = volt_data
    # z = height_data
    plt.pcolormesh(x, y, v, cmap='Greys')
    plt.title("AFM Scan")
    plt.xlabel('x pos [microns]')
    plt.ylabel('y pos [microns]')
    plt.colorbar(label="Z piezo voltage [V]")
    # plt.colorbar(label="Z piezo displacement [nm]")
    plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    print(AFMdata[0])
    # Plot numpy array
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(AFMdata[0])
    fig.colorbar(cax)
    ax.set_title('Piezo voltage image from AFM')
    ax.set_xlabel('x pos [microns]')
    ax.set_ylabel('y pos [microns]')
    plt.show()
