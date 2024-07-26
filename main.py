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


def create_plot(x_data, y_data, title, x_label, y_label, legend_label):
    fig, ax = plt.subplots(1, 1)
    for x, y, label in zip(x_data, y_data, legend_label):
        ax.plot(np.linspace(0, x, len(y)), y, label=label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
    plt.legend()
    plt.show()


def find_edges(images, dx, dy):
    # Initialize list of arrays for sliced data
    slices = []
    widths = []

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

            # Add array and width data to lists
            slices.append(z_x)
            widths.append(width)

    return widths, slices


def get_step_width(scan_width, scan_data):
    step_widths = []
    for width, data in zip(scan_width, scan_data):
        ppm = len(data) / float(width)
        step_width = abs(np.argmax(data) / ppm - np.argmin(data) / ppm)
        step_widths.append(step_width)

    return step_widths


def tilt_correction(scan_width, scan_data):
    sub_arrays = []
    for width, data in zip(scan_width, scan_data):
        ppm = len(data) / float(width)
        if np.argmax(data) < np.argmin(data):
            y = data[:np.argmax(data)]
            x = np.linspace(0, np.argmax(data) / ppm, len(y))
        else:
            y = data[:np.argmin(data)]
            x = np.linspace(0, np.argmin(data) / ppm, len(y))
        # Fit the baseline from the original data
        fit = np.polyfit(x, y, 1)
        linear_baseline = np.poly1d(fit)

        # Subtract the linear baseline from all data points
        sub_data = data - linear_baseline(np.linspace(0, width, len(data)))
        sub_arrays.append(sub_data)

    return sub_arrays


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    includes = ["zoom", "backward"]
    excludes = ['3.5micron', '3.7micron', '3.9micron']
    edge_data = []
    for tup in AFMdata:
        if (all(include in tup[0] for include in includes) and
                all(exclude not in tup[0] for exclude in excludes)):
            # print('\n'.join(str(item) for item in tup))
            create_image(tup[1], tup[0])
            edge_data.append(tup)
    edges = find_edges(edge_data, 0, 0)
    # print('\n'.join(str(item) for item in edges))
    speeds = []
    for tup in edge_data:
        speeds.append(tup[0][4])
    create_plot(edges[0], edges[1], 'Horizontal Edge Resolution', 'x pos [microns]', 'voltage', speeds)
    tilt_corrected = tilt_correction(edges[0], edges[1])
    print(tilt_corrected)
    create_plot(edges[0], tilt_corrected, 'Tilt Corrected', 'x pos [microns]', 'voltage', speeds)
    steps = get_step_width(edges[0], edges[1])
    new_steps = get_step_width(edges[0], tilt_corrected)
    print(steps, new_steps)

