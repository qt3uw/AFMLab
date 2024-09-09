from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


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
    cf_calibrate: float = VOLTS_PER_NM_CF
    ch_calibrate: float = VOLTS_PER_NM_CH

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
        return self

    def get_edge(self, y_coord=None):
        ppm = self.res / self.width
        if y_coord is None:
            y_coord = self.width / 2
        slice_pos = int(y_coord * ppm)
        self.data_slice = self.data[slice_pos, :]

        return self

    def plot_edge(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.linspace(0, self.width, self.res), self.data_slice)
        ax.set_title('Horizontal Slice of ' + self.name)
        ax.set_xlabel('x [microns]')
        ax.set_ylabel('height [nm]')

        return self

    def volt_to_height(self):
        if self.mode.lower() == 'constantforce':
            volts_per_nm = self.cf_calibrate
        else:
            volts_per_nm = self.ch_calibrate

        set_zero = abs(self.data_slice - np.max(self.data_slice))
        self.data_slice = np.divide(set_zero, volts_per_nm)

        return self

    def tilt_correct(self, linear_coords=None):
        ppm = self.res / self.width
        if linear_coords is None:
            start, end = 0, self.res
        else:
            start, end = int(linear_coords[0]*ppm), int(linear_coords[1]*ppm)

        y = self.data_slice[start:end]
        x = np.linspace(0, self.width, len(y))

        A = np.vstack([x, np.ones(len(x))]).T

        # Calculate the linear fit (m: slope, c: intercept)
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        linear_fit = np.interp(np.linspace(0, self.width, self.res), x, m * x + c)
        subtracted_data = self.data_slice - linear_fit
        self.data_slice = subtracted_data - subtracted_data.min()

        return self

    def denoise(self):
        smooth = savgol_filter(self.data_slice, 8, 1)

        return smooth

    def find_features(self):
        ppm = self.res / self.width
        peaks, _ = find_peaks(self.denoise())
        valleys, _ = find_peaks(-self.denoise())

        print('Maxima: ' + str(peaks / ppm), 'Minima: ' + str(valleys / ppm), sep='\n')
        return self

    def get_step_width(self, step_coords):
        """
        :param step_coords: tuple
        :return: float
        """
        ppm = self.res / self.width
        # index of top and bottom of step
        top, bottom = int(step_coords[0]*ppm), int(step_coords[1]*ppm)
        top_val = self.data_slice[top]
        bottom_val = self.data_slice[bottom]
        # nintey and ten percent values
        ninety = (top_val - bottom_val) * 0.9
        ten = (top_val - bottom_val) * 0.1
        # closest index of 90-10 values
        pos_ninety = np.argmin(abs(self.data_slice[min(top, bottom):max(top, bottom)] - ninety))
        pos_ten = np.argmin(abs(self.data_slice[min(top, bottom):max(top, bottom)] - ten))
        step_width = abs(pos_ninety - pos_ten) / ppm

        return step_width

    def get_noise(self, flat_coords):
        """
        :param flat_coords: tuple
        :return: float
        """
        ppm = self.res / self.width
        # index of start and end of flat region
        start, end = int(flat_coords[0]*ppm), int(flat_coords[1]*ppm)
        flat_region = self.data_slice[start:end]
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
    step_heights = []
    for scan in data:
        step_heights.append(np.ptp(scan))

    return np.mean(step_heights)


if __name__ == '__main__':
    myhair = get_scan_data_from_directory('FunScans')
    fig, ax = plt.subplots(1, 1)
    coords = [7.5, 7.5, 4., 4.]
    for scan, coord in zip(myhair, coords):
        scan.plot_afm_image().get_edge(y_coord=coord).volt_to_height().tilt_correct()
        ax.plot(np.linspace(0, scan.width, scan.res), scan.data_slice)
    ax.set_title('Slice of My Hair')
    ax.set_xlabel('x [microns]')
    ax.set_ylabel('height [nm]')
    ax.grid(True)
    plt.show()
