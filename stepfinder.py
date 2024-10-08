from matplotlib import pyplot as plt
import numpy as np

from scandata import get_scan_data_from_directory

# CREATE_STEPS = 0 for first loop, 1 for second loop
CREATE_STEPS = 0

if __name__ == "__main__":
    afmscans = get_scan_data_from_directory('TestSpeed')
    # x coordinates (beginning, end) for each scan
    # TestSpeed folder
    steps = [(0.8704, 0.6868), (1.0584, 0.8496), (1.088, 0.8908), (0.9792, 0.7632), (0.8132, 0.6384), (0.8092, 0.6256),
             (1.1456, 0.9664), (0.8496, 0.6336), (0.9984, 0.6784), (0.7104, 0.4352), (1.5408, 1.296), (1.2528, 1.0728),
             (0.64, 0.4544)]
    back_steps = [(0.816, 0.6324), (0.9936, 0.7848), (0.9996, 0.8092), (0.8784, 0.6768), (0.7828, 0.5244), (0.8364, 0.6256),
                  (1.0688, 0.8512), (0.7344, 0.5328), (0.896, 0.5952), (0.5824, 0.3136), (1.5192, 1.3464), (1.224, 1.044),
                  (0.608, 0.4224)]
    # TestPID folder
    # steps = [(0.576, 0.7168), (0.860, 1.006), (0.815, 1.292), (1.0412, 1.2464)]
    # back_steps = [(0.4608, 0.6016), (0.807, 0.896), (0.6308, 1.1932), (0.9424, 1.1172)]

    if not CREATE_STEPS:
        # find position of step for each scan
        for afmscan in afmscans:
            print(afmscan.backward)
            afmscan.get_edge().volt_to_height().find_features().plot_edge()
            plt.show()
    else:
        # empty lists for plotting
        step_widths, back_step_widths, scan_speeds, back_scan_speeds, scan_rms, back_scan_rms, scan_pid, back_scan_pid \
            = [[] for _ in range(8)]
        # create plots
        fig = plt.figure()
        ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
        ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
        axs = (ax1, ax2, ax3)
        [ax.grid(True) for ax in axs]
        # fill empty lists with data and plot the edge
        for afmscan, step in zip(afmscans, [item for pair in zip(steps, back_steps) for item in pair]):
            afmscan.get_edge().volt_to_height().tilt_correct((0, step[1]))
            # backward scans
            if afmscan.backward:
                back_scan_speeds.append(afmscan.speed)
                back_step_widths.append(afmscan.get_step_width(step))
                back_scan_rms.append(afmscan.get_noise((0, step[1])))
                back_scan_pid.append(afmscan.pid[0])
            # forward scans
            else:
                scan_speeds.append(afmscan.speed)
                step_widths.append(afmscan.get_step_width(step))
                scan_rms.append(afmscan.get_noise((0, step[1])))
                scan_pid.append(afmscan.pid[0])
            # plot slice of edge
            ax1.plot(np.linspace(0, afmscan.width, afmscan.res), afmscan.denoise())

        ax1.set_title('Edge Resolution')
        ax1.set_xlabel('x [microns]')
        ax1.set_ylabel('height [nm]')

        # plot step width against scan speed
        [ax2.scatter(x, y, label=direction) for x, y, direction in
         zip([scan_speeds, back_scan_speeds], [step_widths, back_step_widths], ['forward', 'backward'])]
        ax2.set_title('Step Widths')
        ax2.set_xlabel('scan speeds [pps]')
        ax2.set_ylabel('step width [microns]')
        ax2.legend()
        # plot rms noise against scan speed
        [ax3.scatter(x, y, label=direction) for x, y, direction in
         zip([scan_speeds, back_scan_speeds], [scan_rms, back_scan_rms], ['forward', 'backward'])]
        ax3.set_title('RMS Noise')
        ax3.set_xlabel('scan speeds [pps]')
        ax3.set_ylabel('rms [nm]')
        ax3.legend()
        # # plot step width against P value of PID
        # [ax4.scatter(x, y, label=direction) for x, y, direction in
        #  zip([scan_pid, back_scan_pid], [step_widths, back_step_widths], ['forward', 'backward'])]
        # ax4.set_title('Step Widths')
        # ax4.set_xlabel('proportional value')
        # ax4.set_ylabel('step width [microns]')
        # ax4.legend()
        # # plot rms noise against P value of PID
        # [ax5.scatter(x, y, label=direction) for x, y, direction in
        #  zip([scan_pid, back_scan_pid], [scan_rms, back_scan_rms], ['forward', 'backward'])]
        # ax5.set_title('Step Widths')
        # ax5.set_xlabel('proportional value')
        # ax5.set_ylabel('step width [microns]')
        # ax5.legend()

        plt.show()
