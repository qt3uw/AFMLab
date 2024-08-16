from matplotlib import pyplot as plt
import numpy as np

from oop import get_scan_data_from_directory

# CREATE_STEPS = 0 for first loop, 1 for second loop
CREATE_STEPS = 1

if __name__ == "__main__":
    afmscans = get_scan_data_from_directory('TestSpeed')
    # TestSpeed scans
    steps = [(0.8704, 0.6868), (1.0584, 0.8496), (1.088, 0.8908), (0.9792, 0.7632), (0.8132, 0.6384), (0.8092, 0.6256),
             (1.1456, 0.9664), (0.8496, 0.6336), (0.9984, 0.6784), (0.7104, 0.4352), (1.5408, 1.296), (1.2528, 1.0728),
             (0.64, 0.4544)]
    back_steps = [(0.816, 0.6324), (0.9936, 0.7848), (0.9996, 0.8092), (0.8784, 0.6768), (0.7828, 0.5244), (0.8364, 0.6256),
                  (1.0688, 0.8512), (0.7344, 0.5328), (0.896, 0.5952), (0.5824, 0.3136), (1.5192, 1.3464), (1.224, 1.044),
                  (0.608, 0.4224)]

    if not CREATE_STEPS:
        # Run this loop first to find position of step for each scan
        for afmscan in afmscans:
            print(afmscan.backward)
            afmscan.get_edge().volt_to_height().find_features().plot_edge()
            plt.show()
    else:
        step_widths, back_step_widths, scan_speeds, back_scan_speeds, scan_widths, edges = [[] for _ in range(6)]
        # Run this loop to fill lists for plotting
        i = 0
        for afmscan, step in zip(afmscans, [item for pair in zip(steps, back_steps) for item in pair]):
            print(step)
            afmscan.get_edge().volt_to_height().tilt_correct(end=step[1])
            if not afmscan.backward:
                scan_speeds.append(afmscan.speed)
                step_widths.append(afmscan.get_step_width(step))
            else:
                back_scan_speeds.append(afmscan.speed)
                back_step_widths.append(afmscan.get_step_width(step))
            scan_widths.append(np.linspace(0, afmscan.width, afmscan.res))
            edges.append(afmscan.denoise())

        print(step_widths)
        fig, ax = plt.subplots(2, 1)
        [ax.grid(True) for ax in ax.flat]

        # plot scan of edge
        [ax[0].plot(x, y, label=speed) for x, y, speed in zip(scan_widths, edges, scan_speeds)]
        ax[0].set_title('Edge Resolution')
        ax[0].set_xlabel('x [microns]')
        ax[0].set_ylabel('height [nm]')
        ax[0].legend(title='Scan Speed [pps]')
        # plot step width against scan speed
        [ax[1].scatter(x, y, label=direction) for x, y, direction in
         zip([scan_speeds, back_scan_speeds], [step_widths, back_step_widths], ['forward', 'backward'])]
        ax[1].set_title('Step Widths')
        ax[1].set_xlabel('scan speeds [pps]')
        ax[1].set_ylabel('step width [microns]')
        ax[1].legend()

        plt.show()
