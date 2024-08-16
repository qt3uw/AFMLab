from matplotlib import pyplot as plt
import numpy as np

from oop import get_scan_data_from_directory

# CREATE_STEPS = 0 for first loop, 1 for second loop
CREATE_STEPS = 0

if __name__ == "__main__":
    afmscans = get_scan_data_from_directory('TestSpeed', includes=['backward'])
    # TestSpeed scans
    steps = [(0.816, 0.6324), (0.9936, 0.7848), (0.9996, 0.8092), (0.8784, 0.6768), (0.7828, 0.5244), (0.8364, 0.6256),
             (1.0688, 0.8512), (0.7344, 0.5328), (0.896, 0.5952), (0.5824, 0.3136), (1.5192, 1.3464), (1.224, 1.044),
             (0.608, 0.4224)]

    if not CREATE_STEPS:
        # Run this loop first to find position of step for each scan
        for afmscan in afmscans:
            afmscan.find_edge().volt_to_height().find_step().plot_edge()
            plt.show()
    else:
        step_widths, scan_speeds, scan_widths, edges = [[] for _ in range(4)]
        # Run this loop to fill lists for plotting
        for afmscan, step in zip(afmscans, steps):
            afmscan.find_edge().volt_to_height().tilt_correct(end=step[1])
            step_widths.append(afmscan.get_step_width(step))
            scan_speeds.append(afmscan.speed)
            scan_widths.append(np.linspace(0, afmscan.width, afmscan.res))
            edges.append(afmscan.denoise())

        print(step_widths)
        fig, ax = plt.subplots(2, 1)
        [ax.grid(True) for ax in ax.flat]

        # plot scan of edge
        [ax[0].plot(x, y) for x, y in zip(scan_widths, edges)]
        ax[0].set_title('Edge Resolution')
        ax[0].set_xlabel('x [microns]')
        ax[0].set_ylabel('height [nm]')
        # plot step width against scan speed
        ax[1].scatter(scan_speeds, step_widths)
        ax[1].set_title('Step Widths')
        ax[1].set_xlabel('scan speeds [pps]')
        ax[1].set_ylabel('step width [microns]')

        plt.show()
