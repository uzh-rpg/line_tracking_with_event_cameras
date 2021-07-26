import cv2
import numpy as np
import glob
import yaml
import os

if __name__ == "__main__":

    # read calibration file
    # with open("../../data/kalibr/camera_radtan.yaml") as f:
    # with open("../../data/kalibr/davis_ir_lense_50.yaml") as f:
    # with open("../../data/kalibr/davis_flying_room_rope_no_ir_equi.yaml") as f:
    # with open("../../line_event_tracker/param/flying_arena_real_power_line.yaml") as f:
    # with open("../../line_event_tracker/param/davis_flying_room_rope_ir_equi.yaml") as f:
    with open("../../line_event_tracker/param/davis_flying_room_rope_no_ir_equi.yaml") as f:


    # with open("../../data/kalibr/flying_arena_real_power_line_2.yaml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        # print(data)

    cam = data["cam0"]
    dist_model = cam["distortion_model"]
    intrinsics = cam["intrinsics"]
    resolution = cam["resolution"]
    dist_coeff = np.array(cam["distortion_coeffs"])

    K = np.zeros((3, 3))
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    K[2, 2] = 1

    print(K)

    print(dist_model)
    print(intrinsics)
    print(resolution)
    print(dist_coeff)

    if dist_model == "radtan":
        print("radtan")
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, np.eye(3), K, tuple(resolution), cv2.CV_32FC1)
    elif dist_model == "equidistant":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist_coeff, np.eye(3), K, tuple(resolution), cv2.CV_32FC1)
        print("equi")

    # image_dir = "../../data/flying_room_rope/calibration_ir/images/"
    image_dir = "../../data/flying_room_rope/no_ir/test_recording_2/images/"
    # image_dir = "../../data/flyingarena-20201217T190206Z-002/flyingarena/calibration/images/"
    # image_dir = "../../data/real_power_line/flight2/images/"


    # image_dir_undistorted = "../../data/real_power_line/flight2/images_undistorted_equi/"
    image_dir_undistorted = "../../data/flying_room_rope/no_ir/test_recording_2/images_undistorted_equi/"

    counter = 0
    for filepath in sorted(
            glob.glob(image_dir + "*.png")
    ):

        filename = os.path.basename(filepath)
        print("Image: " + filename)

        image = cv2.imread(filepath)
        image_undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        # # cv2.imwrite("image.png", image)
        print(image_dir_undistorted + filename)
        cv2.imwrite(image_dir_undistorted + filename, image_undistorted)
        counter += 1

        # TESTING using undistort
        # h, w = image.shape[:2]
        # new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 1, (w, h))
        # image_undistorted = cv2.undistort(image, K, dist_coeff, None, new_K)
        # x, y, w, h = roi
        # image_undistorted = image_undistorted[y : y + h, x : x + w]
