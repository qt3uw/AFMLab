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
VOLTS_PER_NM_CF = 0.6363846153846155 / 114
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
    data_slice: np.ndarray = field(default_factory=list)

    def create_from_filepath(self, file_path):
        """
        This method takes an AFM data filepath and modifies the instance of ScanData to store parameter data
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

    def plot_afm_image(self):
        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(self.data, extent=[0, self.width, self.width, 0])
        fig.colorbar(cax)
        ax.set_title('AFM Scan Image')
        ax.set_xlabel('x [microns]')
        ax.set_ylabel('y [microns]')
        # plt.show()

    def find_edge(self, y_coord=None):
        ppm = self.res / self.width
        if y_coord is None:
            slice_pos = self.res / 2
        else:
            slice_pos = y_coord * ppm
        self.data_slice = self.data[int(slice_pos), :]

        return self

    def volt_to_height(self):
        if self.mode.lower() == 'constantforce':
            volts_per_nm = VOLTS_PER_NM_CF
        else:
            volts_per_nm = VOLTS_PER_NM_CH

        set_zero = abs(self.data_slice - np.max(self.data_slice))
        self.data_slice = np.divide(set_zero, volts_per_nm)

        return self

    def tilt_correct(self):
        ppm = self.res / self.width
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(self.data_slice)
        lower = 0.1 * np.max(self.data_slice)
        # Check if step is falling or rising
        if np.argmax(self.data_slice) < np.argmin(self.data_slice):
            linear_region = np.where(self.data_slice > upper)[0][-1] - 10
            y = self.data_slice[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        else:
            linear_region = np.where(self.data_slice < lower)[0][-1] - 10
            y = self.data_slice[:linear_region]
            x = np.linspace(0, linear_region / ppm, len(y))
        # Fit the baseline from the original data
        fit = np.polyfit(x, y, 1)
        linear_baseline = np.poly1d(fit)

        # Subtract the linear baseline from all data points and rescale
        sub_data = self.data_slice - linear_baseline(np.linspace(0, self.width, len(self.data_slice)))
        self.data_slice = rescale_intensity(sub_data, out_range=(self.data_slice.min(), self.data_slice.max()))

        return self

    def denoise(self):
        smooth = savgol_filter(self.data_slice, 8, 1)

        return smooth

    def get_step_width(self):
        ppm = self.res / self.width
        # upper and lower bounds for isolating the step region
        upper = 0.9 * np.max(self.denoise())
        lower = 0.2 * np.max(self.denoise())
        indices = np.where(np.logical_and(self.denoise() < upper, self.denoise() > lower))[0]
        step_width = abs(indices[-1] - indices[0]) / ppm

        return step_width

    def get_noise(self):
        # upper and lower bounds for isolating flat region
        upper = 0.9 * np.max(self.data_slice)
        lower = 0.1 * np.max(self.data_slice)
        # Check if step is falling or rising
        if np.argmax(self.data_slice) < np.argmin(self.data_slice):
            flat_region = self.data_slice[:np.where(self.data_slice > upper)[0][-1] - 10]
        else:
            flat_region = self.data_slice[:np.where(self.data_slice < lower)[0][-1] - 10]

        rms = np.std(flat_region)

        return rms


def get_scan_data_from_directory(folder_name, includes=None, excludes=None):
    """
    This function takes a folder name and creates a ScanData instance for every allowed file the folder
    :param folder_name: String
    :param includes: List of strings
    :param excludes: List of strings
    :return: List of ScanData instances
    """
    folder_path = os.path.join(DATA_DIR, folder_name)

    filepaths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            if (excludes is None and includes is None
                    or excludes is not None and all(exclude not in file_name for exclude in excludes)
                    or includes is not None and all(include in file_name for include in includes)
                    or excludes is not None and includes is not None
                    and all(exclude not in file_name for exclude in excludes)
                    and all(include in file_name for include in includes)):
                filepaths.append(file_path)

    scan_data = []
    for path in filepaths:
        scan = ScanData()
        scan_data.append(scan.create_from_filepath(path))
    return scan_data


def get_step_volts_for_calibration(data):
    """
    This function takes a list of data arrays and calculates the average peak to peak voltage for height calibration
    :param data: numpy array
    :return: float
    """
    # Initialize list for scan heights
    step_heights = []
    for scan in data:
        step_heights.append(np.ptp(scan))

    return np.mean(step_heights)


if __name__ == '__main__':
    afmscans = get_scan_data_from_directory('TestSpeed', excludes=['backward'])
    afmscans_back = get_scan_data_from_directory('TestSpeed', includes=['backward'])

    scan_widths = []
    edge_resolution = []
    tilt_corrected_and_denoised = []
    scan_speeds = []
    scan_speeds_back = []
    step_widths = []
    step_widths_back = []
    rms_noise = []
    rms_noise_back = []
    # step_volts = []
    for afmscan, afmscan_back in zip(afmscans, afmscans_back):
        # step_volts.append(afmscan.find_edge().data_slice)
        scan_widths.append(np.linspace(0, afmscan.width, len(afmscan.data)))
        edge_resolution.append(afmscan.find_edge().volt_to_height().data_slice)
        tilt_corrected_and_denoised.append(afmscan.tilt_correct().denoise())
        scan_speeds.append(afmscan.speed)
        scan_speeds_back.append(afmscan_back.speed)
        step_widths.append(afmscan.get_step_width())
        step_widths_back.append(afmscan_back.find_edge().volt_to_height().tilt_correct().get_step_width())
        rms_noise.append(afmscan.get_noise())
        rms_noise_back.append(afmscan_back.get_noise())
    # print(get_step_volts_for_calibration(step_volts))

    figs, axs = plt.subplots(2, 2)
    for axs in axs.flat:
        axs.grid(True)

    [axs[0, 0].plot(x, y) for x, y in zip(scan_widths, edge_resolution)]
    axs[0, 0].set_title('Edge Resolution')
    # axs[0, 0].legend(title='Scanning Speeds [pps]', fontsize=6)
    [axs[1, 0].plot(x, y) for x, y in zip(scan_widths, tilt_corrected_and_denoised)]
    axs[1, 0].set_title('Tilt corrected and denoised')
    for i in range(2):
        axs[i, 0].set_xlabel('x [microns]')
        axs[i, 0].set_ylabel('height [nm]')
        # axs[i, 0].legend(title='Scanning Speeds [pps]')

    [axs[0, 1].scatter(x, y, label=label) for x, y, label in
        zip([scan_speeds, scan_speeds_back], [step_widths, step_widths_back], ['forward', 'backward'])]
    axs[0, 1].set_title('Step Width')
    axs[0, 1].set_ylabel('Step Widths [microns]')
    [axs[1, 1].scatter(x, y, label=label) for x, y, label in
        zip([scan_speeds, scan_speeds_back], [rms_noise, rms_noise_back], ['forward', 'backward'])]
    axs[1, 1].set_title('RMS Noise')
    axs[1, 1].set_ylabel('RMS [nm]')
    for i in range(2):
        axs[i, 1].set_xlabel('Scanning Speed')
        axs[i, 1].legend(title='Scan Direction')
    plt.show()
