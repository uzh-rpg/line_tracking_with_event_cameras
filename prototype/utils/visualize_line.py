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
path_to_lines = path_to_temp_data + "lines.txt"
path_to_events = path_to_temp_data + "lines_events.txt"


while True:
    lines_df = pd.read_csv(
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

    t = lines_df.iloc[0]["t"]
    image_idx = images_df['t'].sub(t / 1000).abs().idxmin()
    image_name = images_df.iloc[image_idx]['img_name']
    image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)

    lines = []
    for index, row in lines_df.iterrows():

        line_id = int(row["line_id"])
        x_1 = row["x_1"]
        y_1 = row["y_1"]

        x_2 = row["x_2"]
        y_2 = row["y_2"]
        state = int(row["state"])
        lines.append([line_id, x_1, y_1, x_2, y_2, state])


    events = []
    for index, row in events_df.iterrows():

        t = row["t"]
        x = int(row["x"])
        y = int(row["y"])
        p = int(row["p"])

        events.append([x, y, t, p])

    events = np.stack(events, axis=0)
    color = [0.0, 0.0, 1.0]
    image_events = np.zeros(height * width, dtype=np.uint8)
    np.add.at(image_events, (events[:, 0] + events[:, 1] * width).astype("int32"), 1)
    image_events = image_events.reshape((height, width))

    image_events = (
            np.stack(
                [
                    image_events * color[0],
                    image_events * color[1],
                    image_events * color[2],
                    ],
                axis=2,
            )
            * 255
    )

    image_events.astype("uint8")
    print((image_events != 0).shape)
    print(image_frame.shape)
    print(image_name)
    image_frame[image_events != 0] = 0
    image_superimposed = cv2.addWeighted(image_frame, 1, image_events.astype("uint8"), 1, 0)

    plt.imshow(image_superimposed)

    for line in lines:

        if line[5] == 0:
            line_color = "grey"
            text_color = "lightgrey"

        # hibernating lines
        elif line[5] == 1:
            line_color = "sandybrown"
            text_color = "peachpuff"
        # normal lines
        else:
            line_color = "firebrick"
            text_color = "lightcoral"


        end_point_1 = np.array([line[1], line[2]])
        end_point_2 = np.array([line[3], line[4]])
        point = (end_point_1 + end_point_2) / 2
        line_id = line[0]
        plt.text(
            point[0],
            point[1],
            str(line_id),
            fontsize=14,
            fontweight="bold",
            color=text_color,
        )
        plt.plot(
            [end_point_1[0], end_point_2[0]],
            [end_point_1[1], end_point_2[1]],
            linewidth=3,
            color=line_color,
        )
        plt.plot(end_point_1[0], end_point_1[1], color=line_color, linewidth=7)
        plt.plot(end_point_2[0], end_point_2[1], color=line_color, linewidth=7)

    plt.rcParams["figure.figsize"] = (20, 12)
    plt.title("Lines")
    plt.waitforbuttonpress()
    plt.cla()


