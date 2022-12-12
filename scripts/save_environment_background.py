import cv2
import argparse
import numpy as np
import pyrealsense2 as rs
import skimage.exposure
from block_segmentation_utils import *

W = 848
H = 480

# ----- Camera 3 (static) -----
REALSENSE_INTRINSICS_CAM_3 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_3 = "calib/realsense_camera2.tf"
parser = argparse.ArgumentParser()
parser.add_argument(
	"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_CAM_3)
parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_TF_CAM_3)
args = parser.parse_args()
intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
transform = RigidTransform.load(args.extrinsics_file_path)
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322069488')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)

# align stream
aligned_stream = rs.align(rs.stream.color)

# ------ save baseline image -----
frames = pipeline.wait_for_frames()
frames = aligned_stream.process(frames)
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

color_filename = "scripts/Images/color_background.jpg"
