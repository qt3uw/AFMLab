import numpy as np
from matplotlib import pyplot as plt, colors
from pathlib import Path


def get_afm_data(filepath):
    volt_data = np.loadtxt(filepath, dtype=float, delimiter=';')
    keys = ["name", "scan mode", "strain gauge", "resolution", "scan speed"]
    get_string = str(filepath)
    np.loadtxt(get_string, dtype=str, delimiter='_')
    # scan_params = {name: ; mode: ; resolution: }

def create_image(filepath):
    # convert piezo voltage data into an array
    # filepath = Path(r"C:\Users\QT3\Documents\EDUAFM Data\TestSample_ConstantForce_StrainGauge_100px_100pps.csv")
    volt_data = np.loadtxt(filepath, dtype=float, delimiter=';')
    print(volt_data)
    print(np.size(volt_data))
    # convert piezo voltage to height displacement
    # *100 because .Normalize() normalizes between 0 and 1
    # height_data = colors.Normalize()(volt_data) * 100
    # print(height_data)
    # create colormap of data
    x, y = np.mgrid[0:20:0.2, 0:20:0.2]
    v = volt_data
    # z = height_data
    plt.pcolormesh(x, y, v, cmap='Greys')
    plt.title("AFM Scan")
    plt.xlabel('x pos [microns]')
    plt.ylabel('y pos [microns]')
    plt.colorbar(label="Z piezo voltage [V]")
    # plt.colorbar(label="Z piezo displacement [nm]")
    plt.show()
