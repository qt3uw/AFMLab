import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path
import os
import pandas as pd
from skimage import feature, filters

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM\Scans')


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
            # check if file is duplicate
            if str(file_name).split()[0].endswith('.csv'):
                lists = str(file_name).split()[0][:-4].split('_')
            else:
                lists = str(file_name).split()[0].split('_')

            # Append the numpy array and string list to the corresponding list
            array_list.append(array)
            str_list.append(lists)

    # Merge both lists into list of tuples
    tuple_list = list(zip(str_list, array_list))

    return tuple_list


def create_image(data, data_info):
    # Get width of scan from data_info
    width = float(data_info[data_info.index("zoom") + 1][:-6])
    # check if image is a zoomed image
    if "zoom" in data_info:
        extent = [0, width, width, 0]
    else:
        extent = [0, 20, 20, 0]

    # Create color plot of data
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(data, extent=extent)
    fig.colorbar(cax)
    ax.set_title('Piezo voltage image from AFM')
    ax.set_xlabel('x pos [microns]')
    ax.set_ylabel('y pos [microns]')
    # plt.show()


def get_edge(image, image_info, dx, dy):
    # Get width of scan from image_info
    width = float(image_info[image_info.index("zoom") + 1][:-6])
    ppm = np.shape(image)[0] / width

    # Create two plots for horizontal and vertical scans
    fig1, ax1 = plt.subplots(2, 1)
    # sliced horizontal and vertical scans in midpoints of the image
    z_x = image[int(np.shape(image)[0] / 2 + (dx * ppm)), :]
    z_y = image[:, int(np.shape(image)[0] / 2 + (dy * ppm))]
    # horizontal and vertical scales from size of zoomed image
    x = np.linspace(0, width, len(z_x))
    y = np.linspace(width, 0, len(z_y))
    # plot x and y against sliced voltage data
    ax1[0].plot(x, z_x)
    ax1[1].plot(y, z_y)
    ax1[0].text(x[0], z_x[0], "y = " + str(width / 2 + dx))
    ax1[1].text(y[len(y)-1], z_y[len(z_y)-1], "x = " + str(width / 2 + dy))
    ax1[0].set_title('Horizontal Edge Resolution')
    ax1[1].set_title('Vertical Edge Resolution')
    ax1[0].set_xlabel('x pos [microns]')
    ax1[1].set_xlabel('y pos [microns]')
    ax1[0].set_ylabel('Z Piezo Voltage')
    ax1[1].set_ylabel('Z Piezo Voltage')

    plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    param = "zoom"
    for tup in AFMdata:
        if param in tup[0]:
            print('\n'.join(str(item) for item in tup))
            create_image(tup[1], tup[0])
            get_edge(tup[1], tup[0], 0, 0)
