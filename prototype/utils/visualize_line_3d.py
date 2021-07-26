import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# read data
height = 260
width = 346

path_to_temp_data = "../../data/temp/"
path_to_data = "../../data/real_power_line/flight2/"
path_to_lines = path_to_temp_data + "line.txt"
path_to_events = path_to_temp_data + "line_events.txt"


while True:
    line_df = pd.read_csv(
        path_to_lines,
        delim_whitespace=True,
        header=None,
        names=["line_id", "x_1", "y_1", "x_2", "y_2", "t", "state"],
        dtype={"line_id": np.int16, "x_1": np.float64, "y_1": np.float64, "x_2": np.float64, "y_2": np.float64, "t": np.float64, "state": np.int16},
        memory_map=True,
        )

    events_df = pd.read_csv(
        path_to_events,
        delim_whitespace=True,
        header=None,
        names=["t", "x", "y", "p"],
        dtype={"t": np.float64, "x": np.int16, "y": np.int16, "p": np.int16},
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

    t = line_df.iloc[0]["t"]
    image_idx = images_df['t'].sub(t / 1000).abs().idxmin()
    image_name = images_df.iloc[image_idx]['img_name']
    image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)/ 255

    line = line_df.iloc[0]

    # for index, row in line_df.iterrows():
    #
    #     line_id = int(row["line_id"])
    #     print(line_id)
    #     x_1 = row["x_1"]
    #     y_1 = row["y_1"]
    #
    #     x_2 = row["x_2"]
    #     y_2 = row["y_2"]
    #     state = int(row["state"])

    events = []
    for index, row in events_df.iterrows():

        t = row["t"]
        x = int(row["x"])
        y = int(row["y"])
        p = int(row["p"])

        events.append([x, y, t, p])

    events = np.stack(events, axis=0)

    end_point_1 = np.array([line[1], line[2]])
    end_point_2 = np.array([line[3], line[4]])

    x = np.linspace(np.min(events[:, 0]), np.max(events[:, 0]), 5)
    y = np.linspace(np.min(events[:, 1]), np.max(events[:, 1]), 5)
    xx, yy = np.meshgrid(x, y)
    # d = -line.cog.dot(line.normal)
    # z = (-line.normal[0] * xx - line.normal[1] * yy - d) * 1.0 / line.normal[2]

    # fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(events[:, 0], events[:, 1], events[:, 2], c="seagreen", s=15)
    # ax.plot_surface(xx, yy, z, alpha=0.3, color="b")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")

    # max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
    # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
    #         x.max() + x.min()
    # )
    # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
    #         y.max() + y.min()
    # )
    # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
    #         z.max() + z.min()
    # )
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], "w")

    # add image
    image_frame = np.dstack(
        (image_frame, np.full((image_frame.shape[0], image_frame.shape[1]), 0.5))
    )
    x_im = np.arange(0, image_frame.shape[1])
    y_im = np.arange(0, image_frame.shape[0])
    xx_im, yy_im = np.meshgrid(x_im, y_im)
    zz_im = np.full(xx_im.shape, t)
    ax.plot_surface(
        xx_im,
        yy_im,
        zz_im,
        rstride=4,
        cstride=4,
        facecolors=image_frame,
    )

    ax.invert_yaxis()
    ax.view_init(90, 90)
    plt.show()

