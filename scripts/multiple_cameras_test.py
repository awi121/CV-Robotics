import pyrealsense2 as rs
import numpy as np
import cv2
import logging

import argparse
import cv2
import math
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from autolab_core import RigidTransform, YamlConfig

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

# ------ Camera Serial Numbers (necessary to distinguish the two camera streams) -----
# wrist: '220222066259'
# external: '151322061880'

# ------ Configure depth an color streams ------
W = 848
H = 480
# Camera 1 [wrist-mounted]
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('220222066259')
config_1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# Camera 2 [static-mounted]
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('151322061880')
config_2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline_1.start(config_1)
pipeline_2.start(config_2)

# align stream
aligned_stream_1 = rs.align(rs.stream.color)
aligned_stream_2 = rs.align(rs.stream.color)

while True:
	# Camera 1
	frames_1 = pipeline_1.wait_for_frames()
	frames_1 = aligned_stream_1.process(frames_1)
	color_frame_1 = frames_1.get_color_frame()
	depth_frame_1 = frames_1.get_depth_frame().as_depth_frame()
	color_image_1 = np.asanyarray(color_frame_1.get_data())

	# Camera 2
	frames_2 = pipeline_2.wait_for_frames()
	frames_2 = aligned_stream_2.process(frames_2)
	color_frame_2 = frames_2.get_color_frame()
	depth_frame_2 = frames_2.get_depth_frame().as_depth_frame()
	color_image_2 = np.asanyarray(color_frame_2.get_data())

	# Show the images
	cv2.imshow("Wrist Image", color_image_1)
	cv2.imshow("External Image", color_image_2)
	cv2.waitKey(1)