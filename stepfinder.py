from matplotlib import pyplot as plt
import numpy as np

from oop import ScanData, get_scan_data_from_directory, DATA_DIR


if __name__ == "__main__":
    afmscans = get_scan_data_from_directory('TestSpeed', includes=['backward'])
    steps = [(0.816, 0.6324), (0.9936, 0.7848), (0.9996, 0.8092), (0.8784, 0.6768), (0.7828, 0.5244), (0.8364, 0.6256),
             (1.0688, 0.8512), (0.7344, 0.5328), (0.896, 0.5952), (0.5824, 0.3136), (1.5192, 1.3464), (1.224, 1.044),
             (0.608, 0.4224)]
    step_widths, scan_speeds, scan_widths, edges = [[] for _ in range(4)]
    # for afmscan in afmscans:
    #     afmscan.find_edge().volt_to_height().find_step().plot_edge()
    #     plt.show()

    for afmscan, step in zip(afmscans, steps):
        afmscan.find_edge().volt_to_height().tilt_correct(end=step[1])
        step_widths.append(afmscan.get_step_width(step))
        scan_speeds.append(afmscan.speed)
        scan_widths.append(np.linspace(0, afmscan.width, afmscan.res))
        edges.append(afmscan.data_slice)

    print(step_widths)
    fig, ax = plt.subplots(2, 1)
    [ax[0].plot(x, y) for x, y in zip(scan_widths, edges)]
    ax[1].scatter(scan_speeds, step_widths)
    plt.show()
