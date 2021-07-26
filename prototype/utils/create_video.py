import cv2
import numpy as np
import glob
import os

resolution = (346, 260)

out = cv2.VideoWriter(
    "flying_room_rope_recording_2_init.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    40,
    resolution,
    # (511, 389),
)
for filename in sorted(
    glob.glob("../../data/visualizer/flying_room_rope/recording_2/init/*.png"),
    key=os.path.getmtime,
):
    print(filename)
    # img = cv2.imread(filename)[0:resolution(1), 0:resolution(0), :]
    img = cv2.imread(filename)

    out.write(img)
out.release()
