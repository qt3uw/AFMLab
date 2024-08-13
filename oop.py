from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.exposure import rescale_intensity


DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM')
VOLTS_PER_NM_CF = 0.6414615384615386 / 114
VOLTS_PER_NM_CH = 0.10349999999999998 / 114


@dataclass
class ScanData:
    name: str = 'Sample'
    width: float = 20.
    res: int = 250
    speed: int = 100
    mode: str = 'ConstantForce'
    straingauge: bool = False
    pid: tuple = (0.1, 0.1, 0.1)
    lateral: bool = False
    backward: bool = False
    data: np.ndarray = field(default_factory=list)

    def create_from_filepath(self, file_path):
        """
        This method takes an AFM data filepath and modifies the instance of ScanParameters to store parameter data
        :param file_path:
        :return: self
        """

        # Split file name into string list of parameters
        file_name = os.path.basename(file_path)
        file_info = str(file_name).split('.csv')[0].split('_')
        for infostring in file_info:
            for key in self.__dict__:
                if key.lower() in infostring.lower():
                    # case for boolean attributes
                    if key.lower() == infostring.lower():
                        val = not self.__dict__[key]
                    else:
                        val = infostring.lower().replace(key.lower(), '')
                    # case for tuple
                    if isinstance(self.__dict__[key], tuple):
                        val = literal_eval(val)
                    else:
                        val = type(self.__dict__[key])(val)
                    self.__dict__[key] = val

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, delimiter=';', header=None)
            self.data = df.to_numpy()[:, :-1].astype(float)

        return self

    def plot_afm_image(self, data_array=None):
        if data_array is None:
            data_array = self.data
        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(data_array, extent=[0, self.width, self.width, 0])
        fig.colorbar(cax)
        ax.set_title('AFM Scan Image')
        ax.set_xlabel('x [microns]')
        ax.set_ylabel('y [microns]')
        # plt.show()

    def volt_to_height(self, data_array=None):
        if self.mode.lower() == 'constantforce':
            volts_per_nm = VOLTS_PER_NM_CF
        else:
            volts_per_nm = VOLTS_PER_NM_CH

        if data_array is None:
            data_array = self.data
        set_zero = abs(data_array - np.max(data_array))
        to_height = np.divide(set_zero, volts_per_nm)

        return to_height

    def find_edge(self, y_coord=None):
        ppm = self.res / self.width
        if y_coord is None:
            slice_pos = self.res / 2
        else:
            slice_pos = y_coord * ppm
        edge_slice = self.data[int(slice_pos), :]

        return edge_slice

    def tilt_correct(self, data_array):
        ppm = self.res / self.width
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(data_array)
        lower = 0.1 * np.max(data_array)
        # Check if step is falling or rising
        if np.argmax(data_array) < np.argmin(data_array):
            linear_region = np.where(data_array > upper)[0][-1] - 10
            y = data_array[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        else:
            linear_region = np.where(data_array < lower)[0][-1] - 10
            y = data_array[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        # Fit the baseline from the original data
        fit = np.polyfit(x, y, 1)
        linear_baseline = np.poly1d(fit)

        # Subtract the linear baseline from all data points and rescale
        sub_data = data_array - linear_baseline(np.linspace(0, self.width, len(data_array)))
        scaled = rescale_intensity(sub_data, out_range=(data_array.min(), data_array.max()))

        return scaled

    def denoise(self, data_array):
        smooth = savgol_filter(data_array, 8, 1)

        return smooth

    def get_step_width(self, data_array):
        ppm = self.res / self.width
        # upper and lower bounds for isolating the step region
        upper = 0.9 * np.max(data_array)
        lower = 0.2 * np.max(data_array)
        indices = np.where(np.logical_and(data_array < upper, data_array > lower))[0]
        step_width = abs(indices[-1] - indices[0]) / ppm

        return step_width

    def get_noise(self, data_array):
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(data_array)
        lower = 0.1 * np.max(data_array)
        # Check if step is falling or rising
        if np.argmax(data_array) < np.argmin(data_array):
            flat_region = data_array[:np.where(data_array > upper)[0][-1] - 10]
        else:
            flat_region = data_array[:np.where(data_array < lower)[0][-1] - 10]

        rms = np.std(flat_region)

        return rms


def get_scan_data_from_directory(folder_name):
    folder_path = os.path.join(DATA_DIR, folder_name)

    filepaths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            filepaths.append(file_path)

    scan_data = []
    for path in filepaths:
        scan = ScanData()
        scan_data.append(scan.create_from_filepath(path))
    return scan_data


def get_step_height(data):
    # Initialize list for scan heights
    step_heights = []
    for scan in data:
        step_heights.append(np.ptp(scan))

    return np.mean(step_heights)


if __name__ == '__main__':
    afmscans = get_scan_data_from_directory('TestSpeed')

    scan_widths = []
    edge_resolution = []
    tilt_corrected_and_denoised = []
    scan_speeds = []
    step_widths = []
    rms_noise = []
    for afmscan in afmscans:
        scan_widths.append(np.linspace(0, afmscan.width, len(afmscan.data)))
        edge_resolution.append(afmscan.volt_to_height(afmscan.find_edge()))
        tilt_corrected_and_denoised.append(afmscan.denoise(afmscan.tilt_correct(afmscan.volt_to_height(afmscan.find_edge()))))
        scan_speeds.append(afmscan.speed)
        step_widths.append(afmscan.get_step_width(afmscan.denoise(afmscan.tilt_correct(afmscan.volt_to_height(afmscan.find_edge())))))
        rms_noise.append(afmscan.get_noise(afmscan.tilt_correct(afmscan.volt_to_height(afmscan.find_edge()))))

    figs, axs = plt.subplots(2, 2)
    for ax in axs.flat:
        ax.grid(True)

    axs[0, 0].plot(scan_widths[:2], edge_resolution[:2])
    axs[0, 0].set_title('Edge Resolution')
    axs[1, 0].plot(scan_widths, tilt_corrected_and_denoised)
    axs[1, 0].set_title('Tilt corrected and denoised')
    for i in range(2):
        axs[i, 0].set_xlabel('x [microns]')
        axs[i, 0].set_ylabel('height [nm]')
        # axs[i, 0].legend(title='Scanning Speeds [pps]')

    axs[0, 1].scatter(scan_speeds, step_widths)
    axs[0, 1].set_title('Step Width')
    axs[0, 1].set_ylabel('Step Widths')
    axs[1, 1].scatter(scan_speeds, rms_noise)
    axs[1, 1].set_title('RMS Noise')
    axs[1, 1].set_ylabel('RMS [nm]')
    for i in range(2):
        axs[i, 1].set_xlabel('Scanning Speed')
        # axs[i, 1].legend(title='isBackward')

plt.show()
