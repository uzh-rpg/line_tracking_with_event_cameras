import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# read data
height = 260
width = 346

path_to_temp_data = "../../data/temp/"
path_to_data = "../../data/flying_room_rope/no_ir/test_recording_2/"
path_to_lines = path_to_temp_data + "lines.txt"
path_to_events = path_to_temp_data + "events.txt"
path_to_clusters = path_to_temp_data + "clusters.txt"

lines_df = pd.read_csv(
    path_to_lines,
    delim_whitespace=True,
    header=None,
    names=["counter", "t", "state", "n_x", "n_y", "n_t", "c_x", "c_y", "c_t", "m_x", "m_y", "length", "theta", "id"],
    dtype={"counter": np.int16, "t": np.float64, "state": np.int16, "n_x": np.float64, "n_y": np.float64, "n_t": np.float64, "c_x": np.float64, "c_y": np.float64, "c_t": np.float64, "m_x": np.float64, "m_y": np.float64, "length": np.float64, "theta": np.float64, "id": np.int16},
    memory_map=True,
)

clusters_df = pd.read_csv(
    path_to_clusters,
    delim_whitespace=True,
    header=None,
    names=["counter", "t", "n_x", "n_y", "c_x", "c_y"],
    dtype={"counter": np.int16, "t": np.float64, "n_x": np.float64, "n_y": np.float64, "c_x": np.float64, "c_y": np.float64},
    memory_map=True,
)

events_df = pd.read_csv(
    path_to_events,
    delim_whitespace=True,
    header=None,
    names=["counter", "t", "x", "y", "p", "type", "id"],
    dtype={"counter": np.int16, "t": np.float64, "x": np.int16, "y": np.int16, "p": np.int16, "type": np.int16, "id": np.int16},
    memory_map=True,
)

images_df = pd.read_csv(
    path_to_data + "images.txt",
    delim_whitespace=True,
    header=None,
    names=["t", "img_name"],
    dtype={"t": np.float64, "img_name": str},
    memory_map=True,
    )

# reset time
t_min = events_df["t"].min()
events_df["t"] = events_df["t"] - t_min
clusters_df["t"] = clusters_df["t"] - t_min
lines_df["t"] = lines_df["t"] - t_min
images_df["t"] = images_df["t"] - (t_min / 1000)

counters = events_df['counter'].unique()
num_lines = lines_df['id'].max()

colors_idx_shuffeled = np.arange(num_lines)
np.random.shuffle(colors_idx_shuffeled)
colors = cm.rainbow(np.linspace(0, 1, num_lines))
colors = colors[colors_idx_shuffeled]

ax = plt.axes(projection="3d")

for count in counters:
    curr_events_df = events_df[events_df["counter"] == count]
    curr_clusters = clusters_df[clusters_df['counter'] == count]
    curr_lines = lines_df[(lines_df['counter'] == count)]
    curr_t = np.max(curr_events_df["t"].to_numpy())

    image_idx = images_df['t'].sub(curr_t / 1000).abs().idxmin()
    image_name = images_df.iloc[image_idx]['img_name']
    image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)


    for index, line in curr_lines.iterrows():
        line_id = int(line["id"])
        curr_events_line = curr_events_df[(curr_events_df["type"] == 0) & (curr_events_df["id"] == line_id)][["x", "y", "t"]].to_numpy()

        if line["state"] == 2:
            print("Active line")
            color = colors[line_id]
            alpha_events = 0.3
        elif line["state"] == 1:
            print("Hibernating line")
            color = [1.0, 1.0, 0, 1.0]
            alpha_events = 0.3

        elif line["state"] == 0:
            print("Initializing line")
            color = [0.5, 0.5, 0.5, 0.5]
            alpha_events = 0.01

        mid_point_x = line["m_x"]
        mid_point_y = line["m_y"]
        curr_t = line["t"]

        point_1_x = mid_point_x + line["length"] * np.sin(line["theta"]) / 2
        point_1_y = mid_point_y + line["length"] * np.cos(line["theta"]) / 2
        point_2_x = mid_point_x - line["length"] * np.sin(line["theta"]) / 2
        point_2_y = mid_point_y - line["length"] * np.cos(line["theta"]) / 2

        label = "Line: " + str(line_id)
        ax.scatter3D(curr_events_line[:, 2],
                     curr_events_line[:, 0],
                     curr_events_line[:, 1],
                     color=color[:3],
                     alpha=alpha_events,
                     s=10)

        ax.plot([point_1_x, point_2_x],
                [point_1_y, point_2_y],
                zs=curr_t,
                zdir="x",
                label=label,
                color=color,
                linewidth=3)

    # add image
    X = np.arange(0, width)
    Y = np.arange(0, height)
    X, Y = np.meshgrid(X, Y)
    T = np.full(X.shape, curr_t)
    ax.plot_surface(T, X, Y, rstride=1, cstride=1, facecolors=image_frame/255, alpha=0.30)
    ax.view_init(20, -90 * np.cos((curr_t / 1000)))

    ax.legend(bbox_to_anchor=(1,1))
    ax.set_xlim(curr_t - 100, curr_t)
    ax.set_ylim(0, width)
    ax.set_zlim(0, height)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("x [px]")
    ax.set_zlabel("y [px]")
    ax.invert_zaxis()

    plt.savefig(str(count) + ".png", dpi=300)

    # plt.pause(0.1)
    plt.cla()


