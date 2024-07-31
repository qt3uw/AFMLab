import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pandas as pd
from skimage.exposure import rescale_intensity
from scipy.signal import savgol_filter

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


def print_data(data_list, labels):
    if isinstance(labels, list):
        for data, label in zip(data_list, labels):
            neat_format = '{}'.format(label) + '\n' + '\n'.join(str(item) for item in data)
            print(neat_format)
    else:
        neat_format = '\n'.join(str(item) for item in data_list)
        print(neat_format)


def create_image(data_info, data):
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


def create_plot(num_plots, plot_type, x_data, y_data, title, x_label, y_label, leg_label, leg_title):
    fig, ax = plt.subplots(num_plots, 1)
    # One plot case
    if num_plots == 1:
        # Multiple data sets subcase
        if isinstance(leg_label, list):
            for x, y, label in zip(x_data, y_data, leg_label):
                # create array for x_data if value is floating point
                if isinstance(x, float):
                    if plot_type == 'scatter':
                        ax.scatter(np.linspace(0, x, len(y)), y, label=label)
                    elif plot_type == 'line':
                        ax.plot(np.linspace(0, x, len(y)), y, label=label)
                else:
                    if plot_type == 'scatter':
                        ax.scatter(x, y, label=label)
                    elif plot_type == 'line':
                        ax.plot(x, y, label=label)
            plt.legend(title=leg_title)
        # One data set subcase
        else:
            if plot_type == 'scatter':
                ax.scatter(x_data, y_data)
            elif plot_type == 'line':
                ax.plot(x_data, y_data)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
    # Multiple plots case
    else:
        for i in range(num_plots):
            # Multiple data sets per plot subcase
            if isinstance(leg_label, list):
                for x, y, label in zip(x_data, y_data[i], leg_label):
                    # create array for x_data if value is floating point
                    if isinstance(x, float):
                        if plot_type == 'scatter':
                            ax[i].scatter(np.linspace(0, x, len(y)), y, label=label)
                        elif plot_type == 'line':
                            ax[i].plot(np.linspace(0, x, len(y)), y, label=label)
                    else:
                        if plot_type == 'scatter':
                            ax[i].scatter(x, y, label=label)
                        elif plot_type == 'line':
                            ax[i].plot(x, y, label=label)
                    ax[i].legend(title=leg_title)
            # One data set per plot subcase
            else:
                if plot_type == 'scatter':
                    ax[i].scatter(x_data, y_data[i])
                elif plot_type == 'line':
                    ax[i].plot(x_data, y_data[i])
            fig.suptitle(title)
            ax[i].set_xlabel(x_label)
            ax[i].set_ylabel(y_label[i])
            ax[i].grid(True)
    plt.show()


def volt_to_height(volt_data):
    # Initialize list of arrays for converted data
    heights = []

    for item in volt_data:
        # Check if iterating over tuples
        if isinstance(item, tuple):
            to_height = rescale_intensity(item[1], out_range=(0, 114))
        else:
            to_height = rescale_intensity(item, out_range=(0, 114))
        heights.append(to_height)

    return heights


def find_edges(images):
    # Initialize list of arrays for sliced data
    slices = []
    widths = []

    # Loop through all images
    for image in images:
        image_info = image[0]
        image_data = image[1]
        # Only execute find_edge for zoomed images
        if "zoom" not in image_info:
            return
        else:
            # Get width of scan from image_info
            width = float(image_info[image_info.index("zoom") + 1][:-6])
            # ppm = np.shape(image_data)[0] / width

            # sliced horizontal and vertical scans in midpoints of the image
            z_x = image_data[int(np.shape(image_data)[0] / 2), :]
            # z_y = image[1][:, int(np.shape(image[1])[0] / 2)]

            # Add array and width data to lists
            slices.append(z_x)
            widths.append(width)

    return widths, slices


def get_step_width(scan_width, scan_data):
    # Initialize list for step width data
    step_widths = []
    # Loop through data
    for width, data in zip(scan_width, scan_data):
        ppm = len(data) / float(width)
        # upper and lower bounds for isolating the step region
        upper = 0.9 * np.max(data)
        lower = 0.2 * np.max(data)
        indices = np.where(np.logical_and(data < upper, data > lower))[0]
        step_width = abs(indices[-1] - indices[0]) / ppm
        step_widths.append(step_width)

    return step_widths


def get_noise(scan_data):
    # Initialize lists for peak to peak and rms data
    peaks = []
    rms = []
    for data in scan_data:
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(data)
        lower = 0.1 * np.max(data)
        # Check if step is falling or rising
        if np.argmax(data) < np.argmin(data):
            flat_region = data[:np.where(data > upper)[0][-1]-10]
        else:
            flat_region = data[:np.where(data < lower)[0][-1]-10]

        peaks.append(np.ptp(flat_region))
        rms.append(np.std(flat_region))

    return peaks, rms


def denoise(scan_data):
    smoothed = []
    for data in scan_data:
        smooth = savgol_filter(data, 8, 1)
        smoothed.append(smooth)

    return smoothed


def tilt_correction(scan_width, scan_data):
    # Initialize list for tilt corrected data
    sub_arrays = []
    sub_arrays1 = []
    for width, data in zip(scan_width, scan_data):
        ppm = len(data) / float(width)
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(data)
        lower = 0.1 * np.max(data)
        # Check if step is falling or rising
        if np.argmax(data) < np.argmin(data):
            linear_region = np.where(data > upper)[0][-1]-10
            y = data[:linear_region]
            y1 = data[:np.argmax(data)]
            x = np.linspace(0, linear_region / ppm, len(y))
            x1 = np.linspace(0, np.argmax(data) / ppm, len(y1))
        else:
            linear_region = np.where(data < lower)[0][-1]-10
            y = data[:linear_region]
            y1 = data[:np.argmin(data)]
            x = np.linspace(0, linear_region / ppm, len(y))
            x1 = np.linspace(0, np.argmin(data) / ppm, len(y1))
        # Fit the baseline from the original data
        fit = np.polyfit(x, y, 1)
        fit1 = np.polyfit(x1, y1, 1)
        linear_baseline = np.poly1d(fit)
        linear_baseline1 = np.poly1d(fit1)

        # Subtract the linear baseline from all data points and rescale
        sub_data = data - linear_baseline(np.linspace(0, width, len(data)))
        sub_data1 = data - linear_baseline1(np.linspace(0, width, len(data)))
        scaled = rescale_intensity(sub_data, out_range=(0, 114))
        scaled1 = rescale_intensity(sub_data1, out_range=(0, 114))
        sub_arrays.append(scaled)
        sub_arrays1.append(scaled1)

    return sub_arrays, sub_arrays1


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    includes = ['zoom', 'backward']
    excludes = ['3.5micron', '3.7micron', '3.9micron']
    subset = []
    for scan in AFMdata:
        if (all(include in scan[0] for include in includes) and
                all(exclude not in scan[0] for exclude in excludes)):
            # print('\n'.join(str(item) for item in tup))
            subset.append(scan)
    # create_image(subset[0], subset[1])
    # print_data(subset, None)
    height_data = volt_to_height(subset)
    edges = find_edges(subset)
    scan_widths = edges[0]
    # print_data(edges, ['Scan Widths:', 'Scan Data:'])
    height_slice = volt_to_height(edges[1])

    speeds = []
    for sub in subset:
        speeds.append(int(sub[0][4][5:-3]))
    create_plot(1,
                'line',
                scan_widths,
                height_slice,
                'Horizontal Edge Resolution',
                'x pos [microns]',
                'height [nm]',
                speeds,
                'scanning speed [pps]')
    tilt_corrected = tilt_correction(scan_widths, height_slice)
    denoised = denoise(tilt_corrected[0])
    create_plot(2,
                'line',
                scan_widths,
                [tilt_corrected[0], denoised],
                'Tilt Corrected and Denoised',
                'x pos [microns]',
                ['height [nm]', 'height [nm]'],
                speeds,
                'scanning speed [pps]')
    flat = get_noise(tilt_corrected[0])
    # print_data(flat, ['Pk to Pk:', 'RMS'])
    create_plot(2,
                'scatter',
                speeds,
                flat,
                'Pk to Pk vs RMS',
                'Scanning Speed [pixels/s]',
                ['Pk to Pk [nm]', 'RMS [nm]'],
                None,
                None)
    steps = get_step_width(scan_widths, denoised)
    # print('\n'.join(str(item) for item in steps))
    create_plot(1,
                'scatter',
                speeds,
                steps,
                'Step Width',
                'Scanning Speed [pixels/s]',
                'Step Width [microns]',
                None,
                None)
