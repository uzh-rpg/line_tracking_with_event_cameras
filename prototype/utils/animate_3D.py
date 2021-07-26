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

# normalize time
# events_df["t"] = events_df["t"] / events_df["t"].abs().max()

images_df = pd.read_csv(
    path_to_data + "images.txt",
    delim_whitespace=True,
    header=None,
    names=["t", "img_name"],
    dtype={"t": np.float64, "img_name": str},
    memory_map=True,
    )

counters = events_df['counter'].unique()
num_init_lines = lines_df['id'].max()
colors_idx_shuffeled = np.arange(num_init_lines)
np.random.shuffle(colors_idx_shuffeled)
colors = cm.rainbow(np.linspace(0, 1, num_init_lines))
colors = colors[colors_idx_shuffeled]

total = 10

num_whatever = 100

to_plot = [np.random.rand(num_whatever, 3) for i in range(total)]
colors = [np.tile([0,1],num_whatever//2) for i in range(total)]
red_patch = mpatches.Patch(color='red', label='Men')
blue_patch = mpatches.Patch(color='blue', label='Women')

fig = plt.figure()
ax3d = Axes3D(fig)
# scat3D = ax3d.scatter([],[],[], s=10, cmap="bwr", vmin=0, vmax=1)
# scat3D.set_cmap("bwr") # cmap argument above is ignored, so set it manually
scat3D = ax3d.scatter([],[],[], s=10, cmap="rainbow", vmin=0, vmax=1)
scat3D.set_cmap("rainbow")  # cmap argument above is ignored, so set it manually
ttl = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes)

lines = []
for i in range(num_init_lines):
    lines.append(plt.plot(0, 0, 0, lw=4, c='g')[0])

def update_plot(count):

    # set title
    ttl.set_text('PCA on 3 components at step = {}'.format(count*20))

    # set events
    curr_events_df = events_df[events_df["counter"] == count]
    curr_events_line = curr_events_df[(curr_events_df["type"] == 0) & (curr_events_df["id"] != -1)]
    curr_events_line_init = curr_events_df[(curr_events_df["type"] == 0) & (curr_events_df["id"] == -1)]
    curr_events_clusters = curr_events_df[curr_events_df["type"] == 1]
    line_colors = curr_events_line["id"].to_numpy() / num_init_lines
    line_init_colors = np.full((curr_events_line.shape[0],), 0.5)
    cluster_colors = np.full((curr_events_clusters.shape[0],), 0.2)

    curr_events_line_np = curr_events_line[["x", "y", "t"]].to_numpy()
    curr_events_line_init_np = curr_events_line_init[["x", "y", "t"]].to_numpy()
    curr_events_cluster_np = curr_events_clusters[["x", "y", "t"]].to_numpy()

    curr_events = np.vstack((curr_events_line_np, curr_events_line_init_np, curr_events_cluster_np)).T
    curr_colors = np.concatenate((line_colors, line_init_colors, cluster_colors))

    scat3D._offsets3d = curr_events
    scat3D.set_array(curr_colors)

    # set lines and clusters
    curr_lines = lines_df[(lines_df['counter'] == count) & (lines_df["state"] != 0)]
    curr_lines_init = lines_df[(lines_df['counter'] == count) & ((lines_df["state"] == 2) | (lines_df["state"] == 0))]
    curr_clusters = clusters_df[clusters_df['counter'] == count]

    for index, line in curr_lines.iterrows():
        mid_point_x = line["m_x"]
        mid_point_y = line["m_y"]
        mid_point_t = line["t"]

        point_1_x = mid_point_x + line["length"] * np.cos(line["theta"]) / 2
        point_1_y = mid_point_y + line["length"] * np.sin(line["theta"]) / 2
        point_2_x = mid_point_x - line["length"] * np.cos(line["theta"]) / 2
        point_2_y = mid_point_y - line["length"] * np.sin(line["theta"]) / 2

        line_id = int(line["id"])

        lines[line_id].set_data(np.array([point_1_x, point_2_x]), np.array([point_1_y, point_2_y]))
        lines[line_id].set_3d_properties(np.array([mid_point_t, mid_point_t]))

        # ax3d.plot3D((point_1_x, point_2_x), (point_1_y, point_2_y), (mid_point_t, mid_point_t), c=(mid_point_t, mid_point_t))

    ax3d.set_xlim(0, 346)
    ax3d.set_ylim(0, 240)
    ax3d.set_zlim(np.min(curr_events[2, :]), np.max(curr_events[2, :]))

    return scat3D,


def init():
    scat3D.set_offsets([[],[],[]])
    plt.style.use('ggplot')
    plt.legend(handles=[red_patch, blue_patch])


ani = animation.FuncAnimation(fig, update_plot, init_func=init, blit=False, interval=200, frames=counters)
# ani = animation.FuncAnimation(fig, update_plot, init_func=init, blit=False, interval=100, frames=np.arange(total))

# ani.save("ani.gif", writer="imagemagick")

plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# sct = ax.scatter([], [], [], s=10, cmap="bwr", vmin=0, vmax=1)
# sct.set_cmap("bwr")
# title = ax.text2D(0.15, 0.95, "", transform=ax.transAxes)




# def animate(count):
#     # print("Count: " + str(count))
#     curr_events = events_df[events_df['counter'] == count]
#     # curr_lines = lines_df[(lines_df['counter'] == count) & ((lines_df["state"] == 2) | (lines_df["state"] == 1))]
#     # curr_clusters = clusters_df[clusters_df['counter'] == count]
#
#     curr_t = curr_events['t'].max()
#
#     # image_idx = images_df['t'].sub(curr_t / 1000).abs().idxmin()
#     # image_name = images_df.iloc[image_idx]['img_name']
#     # image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)
#
#     # visualize events
#     curr_line_events = curr_events[(curr_events["type"] == 0) & (curr_events["id"] != -1)]
#     # curr_cluster_events = curr_events[curr_events["type"] == 1]
#
#     events_np = curr_line_events[["x", "y", "t"]].to_numpy()
#
#     if events_np.shape[0] > 0:
#         # print("color shape: " + str(colors_np.shape))
#         # print("events shape: " + str(events_np.shape))
#         # print("color min:" + str(colors_np.min()) + ", color max: " + str(colors_np.max()))
#         colors_np = curr_line_events[["id"]].to_numpy()[:, 0] / num_init_lines
#         title.set_text("Events at t: {}".format(curr_t))
#         sct._offset3d = np.transpose(events_np)
#         sct.set_array(colors_np)
#     return sct,
#
#
# def init():
#     sct.set_offsets([[],[],[]])
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('T')
#     # ax.invert_yaxis()
#     # ax.set_xlim(0, 346)
#     # ax.set_ylim(0, 240)
#     plt.style.use('ggplot')
#
#
# anim = animation.FuncAnimation(fig, animate, init_func=init, blit=False, interval=100, frames=counters)
# # anim.save("animation.gif", writer="imagemagick")
# plt.show()





# for count in counters:
#     curr_events = events_df[events_df['counter'] == count]
#     curr_lines = lines_df[(lines_df['counter'] == count) & ((lines_df["state"] == 2) | (lines_df["state"] == 1))]
#     curr_clusters = clusters_df[clusters_df['counter'] == count]
#
#     curr_t = curr_events['t'].max()
#
#     image_idx = images_df['t'].sub(curr_t / 1000).abs().idxmin()
#     image_name = images_df.iloc[image_idx]['img_name']
#     image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)
#
#     # visualize events
#     curr_line_events = curr_events[curr_events["type"] == 0]
#     curr_cluster_events = curr_events[curr_events["type"] == 1]
#
#     num_lines = curr_lines.shape[0]
#     if num_lines > 0:
#         lines = []
#         for index, row in curr_lines.iterrows():
#             # only get id for active lines
#             # if row["state"] == 0:
#             #     continue
#
#             line_events = curr_line_events[curr_line_events["id"] == row["id"]]
#             line_id = int(row["id"])
#             m_x = row["m_x"]
#             m_y = row["m_y"]
#             length = row["length"]
#             theta = row["theta"]
#
#             end_point_1_x = m_x + (length / 2) * np.sin(theta)
#             end_point_1_y = m_y + (length / 2) * np.cos(theta)
#             end_point_2_x = m_x - (length / 2) * np.sin(theta)
#             end_point_2_y = m_y - (length / 2) * np.cos(theta)
#
#             events_np = line_events[["x", "y", "t"]].to_numpy()
#             color = colors[line_id]
#             print(events_np.shape)
#             ax.scatter(events_np[:, 0], events_np[:, 1], events_np[:, 2], color=color, s=25)
#
#             # plt.waitforbuttonpress()
#             # state = int(row["state"])
#             # lines.append([line_id, end_point_1_x, end_point_1_y, end_point_2_x, end_point_2_y, state])
#
#         plt.show()
#
#         # TODO add cluster, add animation, add remaining events