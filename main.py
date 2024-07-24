import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path
import os
import pandas as pd
from skimage import filters as flt
from skimage.morphology import disk

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


def find_edges(images, dx, dy):
    # Initialize list of arrays for sliced data
    slices = []
    widths = []

    # Create two plots for horizontal and vertical scans
    fig1, ax1 = plt.subplots(2, 1)

    # Loop through all images
    for image in images:
        # Only execute find_edge for zoomed images
        if "zoom" not in image[0]:
            return
        else:
            # Get width of scan from image_info
            width = float(image[0][image[0].index("zoom") + 1][:-6])
            ppm = np.shape(image[1])[0] / width

            # sliced horizontal and vertical scans in midpoints of the image
            z_x = image[1][int(np.shape(image[1])[0] / 2 + (dx * ppm)), :]
            # z_y = image[1][:, int(np.shape(image[1])[0] / 2 + (dy * ppm))]
            # horizontal and vertical scales from size of zoomed image
            x = np.linspace(0, width, len(z_x))
            # y = np.linspace(width, 0, len(z_y))

            # plot x and y against sliced voltage data
            ax1[0].plot(x, z_x)
            # ax1[1].plot(y, z_y)
            ax1[0].text(x[int(len(x) / 2)], z_x[int(len(z_x) / 2)]+0.1, str(image[0][4]))
            # ax1[1].text(y[int(len(y) / 2)], z_y[int(len(z_y) / 2)]+0.1, str(image[0][4]))
            ax1[0].set_title('Horizontal Edge Resolution')
            # ax1[1].set_title('Vertical Edge Resolution')
            ax1[0].set_xlabel('x pos [microns]')
            # ax1[1].set_xlabel('y pos [microns]')
            ax1[0].set_ylabel('Z Piezo Voltage')
            # ax1[1].set_ylabel('Z Piezo Voltage')
            [ax1[i].grid(True) for i in range(len(ax1))]

            # Add array and width data to lists
            slices.append(z_x)
            widths.append(width)

    scans = list(zip(slices, widths))

    # plt.show()

    return scans


def get_step_width(scan_data):
    step_widths = []
    for scan in scan_data:
        width = scan[1]
        array = scan[0]
        ppm = len(array) / float(width)
        step_width = abs(np.argmax(array[:int(len(array)/2)]) / ppm - np.argmin(array[:int(len(array)/2)]) / ppm)
        step_widths.append(step_width)

    speeds = [50, 100, 200]
    plt.plot(speeds, step_widths)
    plt.show()
    return step_widths


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    includes = ["zoom", "backward"]
    excludes = []
    edge_images = []
    for tup in AFMdata:
        if (all(include in tup[0] for include in includes) and
                all(exclude not in tup[0] for exclude in excludes)):
            # print('\n'.join(str(item) for item in tup))
            # create_image(tup[1], tup[0])
            edge_images.append(tup)
    edges = find_edges(edge_images, 0, 0)
    # print('\n'.join(str(item) for item in edges))
    # print(type(edges[0][0]))
    steps = get_step_width(edges)
    # print(steps)
