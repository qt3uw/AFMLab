import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path
import os
import pandas as pd
from skimage import feature, filters

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM Data')


def get_afm_data(folder_path):
    # Initialize an empty list to store numpy arrays
    array_list = []
    str_list = []

    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a numpy array
            with open(file_path, 'r') as f:
                df = pd.read_csv(f, delimiter=';', header=None)
                array = df.to_numpy()[:, :-1].astype(float)

            # Convert the file_name into list of identifiers
            if str(file_name)[:-4].endswith(')'):
                lists = str(file_name)[:-8].split('_')
            else:
                lists = str(file_name)[:-4].split('_')

            # Append the numpy array and string list to the corresponding list
            array_list.append(array)
            str_list.append(lists)

    # Merge both lists into list of tuples
    tuple_list = list(zip(str_list, array_list))

    return tuple_list


def create_image(data):
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(data, extent=[0, 20, 0, 20])
    fig.colorbar(cax)
    ax.set_title('Piezo voltage image from AFM')
    ax.set_xlabel('x pos [microns]')
    ax.set_ylabel('y pos [microns]')
    plt.show()


def get_edge(image):
    canny = feature.canny(image, sigma=2.6)
    farid = filters.farid(image)
    laplace = filters.laplace(image)
    prewitt = filters.prewitt(image)
    roberts = filters.roberts(image)
    scharr = filters.scharr(image)
    edges = [canny, farid, laplace, prewitt, roberts, scharr]
    fig, ax = plt.subplots(1, 6)
    i = 0
    for edge in edges:
        ax[i].imshow(edge, extent=[0, 20, 0, 20])
        ax[i].set_title('Edges of AFM Scan')
        ax[i].set_xlabel('x pos [microns]')
        ax[i].set_ylabel('y pos [microns]')
        i = i + 1
    plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    for tup in AFMdata[16:17]:
        if tup[0][2] == 'StrainGauge':
            print('\n'.join(str(item) for item in tup))
            # create_image(tup[1])
            get_edge(tup[1])
