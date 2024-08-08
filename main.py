import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pandas as pd
from skimage.exposure import rescale_intensity
from scipy.signal import savgol_filter

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM\Scans')
PARAM_HEADERS = ['Name', 'Resolution', 'Speed', 'Mode', 'StrainGauge', 'is_Zoom', 'Width',
                 'PID', 'P', 'I', 'D', 'is_Lateral', 'Direction']
VOLTS_PER_NM_CF = 0.6414615384615386 / 114
VOLTS_PER_NM_CH = 0.10349999999999998 / 114


# Arguments: folder_path: PATH
# Returns: tuple list. The first element is a list of strings and second is an array
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
                file_info = str(file_name).split()[0][:-4].split('_')
            else:
                file_info = str(file_name).split()[0].split('_')
            # # check if optional parameters are in file info
            # for i, delta in zip(range(4, 9), PARAM_HEADERS[4:]):
            #     if delta not in file_info:
            #         file_info.insert(i, 'Default/None')

            # Append the numpy array and string list to the corresponding list
            array_list.append(array)
            str_list.append(file_info)

    # Merge both lists into list of tuples
    tuple_list = list(zip(str_list, array_list))

    return tuple_list


# Arguments: data: tuple list, includes: str list, excludes: str list
# Returns: tuple list
def filter_data(data, includes=None, excludes=None):
    # Initialize list for filtered data
    subset = []
    for scan in data:
        data_info = scan[0]
        # check which parameters to include and exclude from subset
        if not isinstance(includes, list):
            if all(exclude not in data_info for exclude in excludes):
                subset.append(scan)
        elif not (isinstance(excludes, list)):
            if all(include in data_info for include in includes):
                subset.append(scan)
        else:
            if (all(include in data_info for include in includes) and
                    all(exclude not in data_info for exclude in excludes)):
                subset.append(scan)

    return subset


# Arguments: data: tuple list, param: str, as_type: str, begin_slice: int, end_slice: int
# Returns: pandas dataframe, str list
# Notes: param must be an element of PARAM_HEADERS, as_type can be 'float' or 'int' (otherwise assume str)
def get_parameter(data, param=None, as_type=None, begin_slice=None, end_slice=None):
    # Initialize lists for dataframe and parameter of interest
    info_list = []
    get_param = []

    for scan in data:
        # Initialize list to hold scan info
        data_info = []
        data_info.extend(scan[0])
        # make resolution and speed parameters more readable
        data_info[1] = data_info[1][3:]
        data_info[2] = data_info[2][5:]
        # add default values if list has missing parameters to match PARAM_HEADERS
        if 'backward' not in data_info:
            data_info.append('forward')
        if 'Lateral' not in data_info:
            data_info.insert(-1, None)
        if 'PID' not in data_info:
            data_info.insert(-2, 'Default')
            data_info.insert(-2, '0.1')
            data_info.insert(-2, '0.1')
            data_info.insert(-2, '0.1')
        if 'zoom' not in data_info:
            data_info.insert(-6, None)
            data_info.insert(-6, None)
        if 'StrainGauge' not in data_info:
            data_info.insert(-8, 'Off')
        info_list.append(data_info)

    # Convert list into dataframe
    all_param = pd.DataFrame(info_list, columns=PARAM_HEADERS)

    # check if fetching parameter or just returning the dataframe
    if not isinstance(param, str):

        return all_param.to_string()

    else:
        # fetch one parameter from dataframe
        get_col = all_param[param].tolist()
        # check what type of parameter and if slicing is necessary
        for item in get_col:
            if as_type == 'int':
                part = int(item[begin_slice:end_slice])
            elif as_type == 'float':
                part = float(item[begin_slice:end_slice])
            else:
                part = item[begin_slice:end_slice]
            get_param.append(part)

        return get_param


# Arguments: volt_data: tuple list or arr list
# Returns: same type
def volt_to_height(volt_data, mode):
    # Initialize list of arrays for converted data
    heights = []
    if mode == 'ConstantForce':
        volts_per_nm = VOLTS_PER_NM_CF
    else:
        volts_per_nm = VOLTS_PER_NM_CH

    for scan in volt_data:
        # Check if iterating over tuples
        if isinstance(scan, tuple):
            set_zero = abs(scan[1] - np.max(scan[1]))
            to_height = np.divide(set_zero, volts_per_nm)
        else:
            set_zero = abs(scan - np.max(scan))
            to_height = np.divide(set_zero, volts_per_nm)
        heights.append(to_height)

    return heights


# Arguments: images: tuple list
# Returns: list tuple
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


# Arguments: scan_width: list, scan_data: arr list
# Returns: arr list
def tilt_correction(scan_width, scan_data):
    # Initialize list for tilt corrected data
    sub_arrays = []

    for width, data in zip(scan_width, scan_data):
        ppm = len(data) / float(width)
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(data)
        lower = 0.1 * np.max(data)
        # Check if step is falling or rising
        if np.argmax(data) < np.argmin(data):
            linear_region = np.where(data > upper)[0][-1]-10
            y = data[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        else:
            linear_region = np.where(data < lower)[0][-1]-10
            y = data[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        # Fit the baseline from the original data
        fit = np.polyfit(x, y, 1)
        linear_baseline = np.poly1d(fit)

        # Subtract the linear baseline from all data points and rescale
        sub_data = data - linear_baseline(np.linspace(0, width, len(data)))
        scaled = rescale_intensity(sub_data, out_range=(data.min(), data.max()))
        sub_arrays.append(scaled)

    return sub_arrays


# Arguments: scan_data: arr list
# Returns: same type
def denoise(scan_data):
    smoothed = []
    for data in scan_data:
        smooth = savgol_filter(data, 8, 1)
        smoothed.append(smooth)

    return smoothed


# Arguments: data: arr list
# Returns: float
def get_step_height(data):
    # Initialize list for scan heights
    step_heights = []
    for scan in data:
        step_heights.append(np.ptp(scan))

    return np.mean(step_heights)


# Arguments: scan_width: list, scan_data: arr list
# Returns: list
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


# Arguments: scan_data: arr list
# Returns: list tuple
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


# Arguments: data: any type
# Returns: None
def print_data(data):
    if isinstance(data, list) and all(isinstance(element, (tuple, list, np.ndarray)) for element in data):
        neat_format = '\n'.join(str(item) for item in data)
        print(neat_format)
    else:
        print(data)


# Arguments: data_info: str list, data: arr list, title: str, x_label: str, y_label: str
# Returns: None
def create_image(images, titles, x_label, y_label):
    for image, title in zip(images, titles):
        image_info = image[0]
        image_data = image[1]
        # Get width of scan from data_info
        width = float(image_info[image_info.index("zoom") + 1][:-6])
        # check if image is a zoomed image
        if "zoom" in image_info:
            extent = [0, width, width, 0]
        else:
            extent = [0, 20, 20, 0]

        # Create color plot of data
        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(image_data, extent=extent)
        fig.colorbar(cax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # plt.show()


# Arguments: num_plots: int, plot_type: str, x_data: list, y_data: list, title: str, x_label: str, y_label: str,
# leg_label: str list, leg_title: str
# Returns: None
# Notes: plot_type can be 'line' or 'scatter', all plots must have the same x_data but can have different y_data
def create_plot(num_plots, plot_type, x_data, y_data, title, x_label, y_label, leg_label=None, leg_title=None):
    fig, ax = plt.subplots(num_plots, 1)
    # One plot case
    if num_plots == 1:
        # Multiple data sets subcase
        if len(np.shape(y_data)) > 1:
            if not isinstance(leg_label, list):
                raise ValueError("Make a legend!")
            for x, y, label in zip(x_data, y_data, leg_label):
                # create array for x_data if value is floating point
                if isinstance(x, (int, float)):
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
            if isinstance(x_data, (int, float)):
                if plot_type == 'scatter':
                    ax.scatter(np.linspace(0, x_data, len(y_data)), y_data)
                elif plot_type == 'line':
                    ax.plot(np.linspace(0, x_data, len(y_data)), y_data)
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
            if len(np.shape(y_data[i])) > 1:
                if not isinstance(leg_label, list):
                    raise ValueError("Make a legend!")
                for x, y, label in zip(x_data, y_data[i], leg_label):
                    # create array for x_data if value is floating point
                    if isinstance(x, (int, float)):
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
                if isinstance(x_data, (int, float)):
                    if plot_type == 'scatter':
                        ax[i].scatter(np.linspace(0, x_data, len(y_data[i])), y_data[i])
                    elif plot_type == 'line':
                        ax[i].plot(np.linspace(0, x_data, len(y_data[i])), y_data[i])
                else:
                    if plot_type == 'scatter':
                        ax[i].scatter(x_data, y_data[i])
                    elif plot_type == 'line':
                        ax[i].plot(x_data, y_data[i])
            fig.suptitle(title)
            ax[i].set_xlabel(x_label)
            ax[i].set_ylabel(y_label[i])
            ax[i].grid(True)
    # plt.show()


if __name__ == '__main__':
    AFMdata = get_afm_data(DATA_DIR)
    zoom_images = filter_data(AFMdata, ['zoom', 'backward'], ['3.5micron', '3.7micron', '3.9micron'])
    constant_force = filter_data(zoom_images, ['ConstantForce'])
    constant_height = filter_data(zoom_images, ['InvertedCircles', 'ConstantHeight'], ['Lateral'])
    lateral_force = filter_data(zoom_images, ['InvertedCircles', 'Lateral'])
    cf_edges = find_edges(constant_force)
    cf_widths = cf_edges[0]
    cf_volts = cf_edges[1]
    ch_edges = find_edges(constant_height)
    ch_widths = ch_edges[0]
    ch_volts = ch_edges[1]
    lateral_edges = find_edges(lateral_force)
    lateral_widths = lateral_edges[0]
    lateral_volts = lateral_edges[1]
    cf_heights = volt_to_height(cf_volts, 'ConstantForce')
    ch_heights = volt_to_height(ch_volts, 'ConstantHeight')
    cf_tilt_corrected = tilt_correction(cf_widths, cf_heights)
    ch_tilt_corrected = tilt_correction(ch_widths, ch_heights)
    cf_denoised = denoise(cf_tilt_corrected)
    cf_flat = get_noise(cf_tilt_corrected)
    cf_steps = get_step_width(cf_widths, cf_denoised)
    cf_speeds = get_parameter(constant_force, 'Speed', 'int', 0, -3)
    ch_names = get_parameter(lateral_force, 'Name')
    print_data(ch_volts)
    ch_params = get_parameter(constant_height)
    print_data(ch_params)
    create_plot(2,
                'line',
                ch_widths,
                [ch_heights, lateral_volts],
                'Constant Height',
                'x pos [micron]',
                ['height [nm]', 'y deflection voltage'],
                ch_names)
    # plt.show()
    create_plot(2,
                'line',
                cf_widths,
                [cf_heights, cf_volts],
                'Horizontal Edge Resolution',
                'x pos [microns]',
                ['height [nm]', 'z piezo voltage'],
                cf_speeds,
                'scanning speed [pps]')
    create_plot(2,
                'line',
                cf_widths,
                [cf_tilt_corrected, cf_denoised],
                'Tilt Corrected and Denoised Height',
                'x pos [microns]',
                ['height [nm]', 'height [nm]'],
                cf_speeds,
                'scanning speed [pps]')
    create_plot(2,
                'scatter',
                cf_speeds,
                cf_flat,
                'Pk to Pk vs RMS',
                'Scanning Speed [pixels/s]',
                ['Pk to Pk [nm]', 'RMS [nm]'])
    create_plot(1,
                'scatter',
                cf_speeds,
                cf_steps,
                'Step Width',
                'Scanning Speed [pixels/s]',
                'Step Width [microns]')
