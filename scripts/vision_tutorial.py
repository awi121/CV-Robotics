from frankapy import FrankaArm
import numpy as np
import argparse
import apriltag
from pupil_apriltags import Detector
import cv2
import math
import random
import threading
import queue
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
import pyrealsense2 as rs
from DetectObject import DetectObject

from perception import CameraIntrinsics
from utils import *

from autolab_core import RigidTransform, YamlConfig



W = 848
H = 480

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

while True:
# for i in range(50):
	# ----- added from other method
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()

	points = point_cloud.calculate(depth_frame)
	verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
	
	# Convert images to numpy arrays
	depth_image = np.asanyarray(depth_frame.get_data())

	# skip empty frames
	if not np.any(depth_image):
		print("no depth")
		# continue

	# print("\n[INFO] found a valid depth frame")
	color_image = np.asanyarray(color_frame.get_data())
	bw_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	# Show the images
	cv2.imshow("Image", color_image)
	cv2.waitKey(1)