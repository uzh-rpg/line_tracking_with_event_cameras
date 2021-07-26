import math

import numpy as np
import pandas as pd

from line_tracker import LineTracker
from utils.config import set_static_variables, config

if __name__ == "__main__":

    # read data
    # path_to_data = "data/slider_depth/"
    # path_to_data = "data/shapes_translation/"

    # path_to_data = "data/alex_tests/davis_camera/camera_6dof_with_carpet/camera_6dof_with_carpet_slow/"
    # path_to_data = (
    #     "data/alex_tests/davis_camera/camera_6dof_line/camera_6dof_crossing_line/"
    # )
    # path_to_data = "../data/flyingarena-20201217T190206Z-002/flyingarena/flight1/"
    path_to_data = "../data/real_power_line/flight2/"

    # path_to_data = "data/alex_tests/davis_camera/camera_translation_static_disc/camera_translation_rot_disc/"
    # path_to_data = "data/urban/"
    config["path_to_data"] = path_to_data
    events_df = pd.read_csv(
        path_to_data + "events.txt",
        delim_whitespace=True,
        header=None,
        names=["t", "x", "y", "p"],
        dtype={"t": np.float64, "x": np.int16, "y": np.int16, "p": np.int16},
        nrows=config["number_of_events_testing"],
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

    events_df = events_df[events_df.t > config["start_time"]]
    images_df = images_df[images_df.t > config["start_time"]]

    set_static_variables(config)
    line_tracker = LineTracker(images_df, config)

    for index, row in events_df.iterrows():

        t = row["t"] * config["time_scaling_factor"]
        x = int(row["x"])
        y = int(row["y"])
        p = int(row["p"])

        line_tracker.update(x, y, t, p)
