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
path_to_chain = path_to_temp_data + "chain.txt"

while True:
    chain_df = pd.read_csv(
        path_to_chain,
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

    events = []
    for index, row in chain_df.iterrows():

        t = row["t"]
        x = int(row["x"])
        y = int(row["y"])
        p = int(row["p"])

        events.append([x, y, t, p])

    events = np.stack(events, axis=0)

    image_idx = images_df['t'].sub(events[-1, 2] / 1000).abs().idxmin()

    image_name = images_df.iloc[image_idx]['img_name']


    # color = [203.0 / 255, 255.0 / 255, 229.0 / 255]
    color = [255.0 / 255.0, 0.0 , 0.0]
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

    image_frame = cv2.cvtColor(cv2.imread(path_to_data + image_name, 0), cv2.COLOR_GRAY2RGB)

    image_frame[image_events != 0] = 0
    image_superimposed = cv2.addWeighted(image_frame, 1, image_events.astype("uint8"), 1, 0)


    plt.rcParams["figure.figsize"] = (20, 12)
    plt.imshow(image_superimposed)
    plt.waitforbuttonpress()
    plt.cla()


