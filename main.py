import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path
import os
import pandas as pd

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

            # Convert file_name into list of identifiers
            if str(file_name)[:-4].endswith(')'):
                lists = str(file_name)[:-8].split('_')
            else:
                lists = str(file_name)[:-4].split('_')

            # Append the numpy array to the list
            array_list.append(array)
            str_list.append(lists)

            # Merge into list of tuples
    tuple_list = list(zip(str_list, array_list))

    return tuple_list


def create_image(data):
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(data)
    fig.colorbar(cax)
    ax.set_title('Piezo voltage image from AFM')
    ax.set_xlabel('x pos [microns]')
    ax.set_ylabel('y pos [microns]')
    plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    print(AFMdata[0])
    create_image(AFMdata[0][1])
