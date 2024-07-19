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
    # plt.show()


def get_edge(image):
    # Different edge detection operators
    canny = feature.canny(image, sigma=2.6)
    farid = filters.farid(image)
    laplace = filters.laplace(image)
    prewitt = filters.prewitt(image)
    roberts = filters.roberts(image)
    scharr = filters.scharr(image)

    # Plot each edge detection of the image
    # edges = [canny, farid, laplace, prewitt, roberts, scharr]
    # fig, ax = plt.subplots(1, 6)
    # i = 0
    # for edge in edges:
    #     ax[i].imshow(edge, extent=[0, 20, 0, 20])
    #     ax[i].set_title('Edges of AFM Scan')
    #     ax[i].set_xlabel('x pos [microns]')
    #     ax[i].set_ylabel('y pos [microns]')
    #     i = i + 1

    # Lists of coordinates of edges
    left = [(3.5, 16.7),
            (10., 16.8),
            (6.8, 13.4),
            (13.5, 13.7),
            (10.2, 10.3),
            (3.5, 10.),
            (6.8, 6.8),
            (13.5, 17.1),
            (10.1, 3.7),
            (3.6, 3.4)]
    top = [(4.8, 17.9),
           (11.5, 18.2),
           (8.1, 14.8),
           (14.9, 15.),
           (11.5, 11.6),
           (5., 11.4),
           (8.1, 8.2),
           (15, 8.5),
           (11.6, 5),
           (4.9, 4.8)]
    right = [(6.2, 16.7),
             (12.8, 16.8),
             (9.4, 13.4),
             (16.3, 13.7),
             (12.8, 10.3),
             (6.2, 10.),
             (9.5, 6.8),
             (16.3, 17.1),
             (12.9, 3.7),
             (6.2, 3.4)]
    bottom = [(4.8, 15.2),
              (11.5, 15.5),
              (8.1, 12.2),
              (14.9, 12.4),
              (11.5, 9),
              (5., 8.6),
              (8.1, 5.6),
              (15., 5.8),
              (11.6, 2.4),
              (4.9, 2.1)]

    fig1, ax1 = plt.subplots(2, 1)
    ppm = np.size(image[0]) / 20
    # for loop to plot data points on top of each other
    for i in range(len(left)):
        # convert micron measurements into array index
        z_x_row = int(left[i][1]*ppm)
        z_x_start = int(left[i][0]*ppm)
        z_x_end = int(right[i][0]*ppm)
        z_y_col = int(bottom[i][0]*ppm)
        z_y_start = int(bottom[i][1]*ppm)
        z_y_end = int(top[i][1]*ppm)
        # sliced horizontal and vertical scans of data
        z_x = image[z_x_row, z_x_start:z_x_end]
        z_y = image[z_y_col, z_y_start:z_y_end]
        # horizontal and vertical scales from measured edge coordinates
        x = np.linspace(left[i][0] - 2, right[i][0] + 2, len(z_x))
        y = np.linspace(bottom[i][1] - 2, top[i][1] + 2, len(z_y))
        # plot x and y against sliced voltage data
        ax1[0].plot(x, z_x)
        ax1[1].plot(y, z_y)
        ax1[0].text(x[0],z_x[0]+0.2, "y = " + str(left[i][1]))
        ax1[1].text(y[0],z_y[0]+0.2, "x = " + str(bottom[i][0]))
        ax1[0].set_title('Horizontal Edge Resolution')
        ax1[1].set_title('Vertical Edge Resolution')
        ax1[0].set_xlabel('x pos [microns]')
        ax1[1].set_xlabel('y pos [microns]')
        # ax1[0].set_ylabel('Z Piezo Voltage')

    plt.show()


def get_blob(image):
    dog = feature.blob_dog(image, min_sigma=20, max_sigma=50, threshold=0.001)
    doh = feature.blob_doh(image, min_sigma=20, max_sigma=50, threshold=0.001)
    log = feature.blob_log(image, min_sigma=20, max_sigma=50, num_sigma=10, threshold=0.001)
    blobs = [dog, doh, log]
    fig, ax = plt.subplots(1, 3)
    i = 0
    for blob in blobs:
        ax[i].imshow(blob, extent=[0, 20, 0, 20])
        ax[i].set_title('Blobs of AFM Scan')
        ax[i].set_xlabel('x pos [microns]')
        ax[i].set_ylabel('y pos [microns]')
        i = i + 1
    plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    for tup in AFMdata[16:17]:
        if tup[0][2] == 'StrainGauge':
            print('\n'.join(str(item) for item in tup))
            create_image(tup[1])
            get_edge(tup[1])
