from dataclasses import dataclass, field
from pathlib import Path
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# Example most complicated filename
# sample_resolution256_speed100_modeconstantforce_stra

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM\Scans')


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
                    if key.lower() == infostring.lower():
                        val = not self.__dict__[key]
                    else:
                        val = infostring.lower().replace(key.lower(), '')
                    val = type(self.__dict__[key])(val)
                    self.__dict__[key] = val

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, delimiter=';', header=None)
            self.data = df.to_numpy()[:, :-1].astype(float)

        return self

    def plot_afm_image(self):
        pass

    def get_step_width(self):
        step_width = None  # code here
        return step_width

        # Code here to put the right number from infostring into self.__dict__[key] =


def get_scan_data_from_directory(folder_path):
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


if __name__ == '__main__':
    # p = ScanData()
    # p = p.create_from_filepath(DATA_DIR)
    # print(p.Name)
    # p.Name = 'changed name'
    # print(p.Name)
    # print(p.__dict__)
    afmscans = get_scan_data_from_directory(DATA_DIR)

    # figs, axs = plt.subplots(1, 1)
    for afmscan in afmscans:
        print(type(afmscan))
        print(afmscan.__dict__)
    #     axs.plot(scans.speed, scans.get_step_width(), marker='o', linestyle='None', color='k')
