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
# from perception_utils.apriltags import AprilTagDetector
# from perception_utils.realsense import get_first_realsense_sensor

# REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
# REALSENSE_EE_TF = "calib/realsense_ee.tf"

REALSENSE_INTRINSICS = "calib/realsense_static_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_static.tf"

def vision_loop(realsense_intrinsics, realsense_to_ee_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	# config.enable_device('220222066259')
	config.enable_device('151322061880')
	config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	# print("[INFO] start streaming...")
	pipeline.start(config)

	aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud = rs.pointcloud()

	# get a stream of images
	while True:
	# for i in range(50):
		# ----- added from other method
		current_pose = fa.get_pose()
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

		# python wrapper AprilTag package
		detector = Detector(families="tag36h11",
			nthreads=1,
			quad_decimate=1.0,
			quad_sigma=0.0,
			refine_edges=1,
			decode_sharpening=0.25,
			debug=0)

		# camera parameters [fx, fy, cx, cy]
		cam_param = [realsense_intrinsics.fx, realsense_intrinsics.fy, realsense_intrinsics.cx, realsense_intrinsics.cy]
		detections = detector.detect(bw_image, estimate_tag_pose=True, camera_params=cam_param, tag_size=0.03)
		# print("\nNumber of AprilTags: ", len(detections))


		# ------- SECOND CAMERA NOTES -------
		# loop over the detections from both cameras (some kind of list of tuples?)
		# where if both cameras detected the same tag, then do everything else the same?





		# loop over the detected AprilTags
		for d in detections:

			# check if apriltag has been detected before
			obj_id = d.tag_id
			if obj_id not in detected_objects:
				# add tag to the dictionary of detected objects
				tagFamily = d.tag_family.decode("utf-8")
				detected_objects.update({obj_id : DetectObject(object_id=obj_id, object_class=tagFamily)})

			# extract the bounding box (x, y)-coordinates for the AprilTag
			# and convert each of the (x, y)-coordinate pairs to integers
			(ptA, ptB, ptC, ptD) = d.corners
			ptB = (int(ptB[0]), int(ptB[1]))
			ptC = (int(ptC[0]), int(ptC[1]))
			ptD = (int(ptD[0]), int(ptD[1]))
			ptA = (int(ptA[0]), int(ptA[1]))

			# draw the bounding box of the AprilTag detection
			cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
			cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
			cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
			cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)

			# draw the center (x, y)-coordinates of the AprilTag
			(cX, cY) = (int(d.center[0]), int(d.center[1]))
			cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

			# --------- added code to calculate AprilTag x,y,z position ------
			bounds = np.array([ptA, ptB, ptC, ptD])
			obj = detected_objects[obj_id]
			translation_matrix = d.pose_t
			translation_matrix = np.array(translation_matrix).reshape(3)
			rotation_matrix = d.pose_R
			pose_error = d.pose_err
			object_center_point = obj.get_position_apriltag(bounds, verts, current_pose, translation_matrix)

			# print("\nTranslation: ", translation_matrix)

			string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
			cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# put updated dictionary in queue for other thread to access
		object_queue.put(detected_objects)

		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)

def position_loop(object_queue):
	while True:
		detected_objects = object_queue.get()
		for obj_id in detected_objects:
			pass
			# print("\nObject ID: ", obj_id, " Current Position: ", detected_objects[obj_id]._return_current_position())
			# print("\n")
			# velocity = detected_objects[obj_id]._return_current_velocity()
			# rotation = detected_objects[obj_id]._return_current_rotation()
			# ang_velocity = detected_objects[obj_id]._return_current_ang_velocity()

			# color = detected_objects[obj_id]._return_color()
			# type = detected_objects[obj_id]._return_type()
			# size = detected_objects[obj_id]._return_size()

if __name__ == "__main__":
	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	# parser.add_argument("--extrinsics_static_file_path", type=str, default=REALSENSE_STATIC_TF)
	args = parser.parse_args()
	
	# reset pose and joints
	fa = FrankaArm()
	fa.reset_pose()
	fa.reset_joints()

	# # move to center
	# pose = fa.get_pose()
	# print("\nRobot Pose: ", pose)
	# pose.translation = np.array([0.6, 0, 0.5])
	# fa.goto_pose(pose)

	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)
	# realsense_to_static_transform = RigidTransform.load(args.extrinsics_static_file_path)

	print("\nCamera Intrinsics: ", realsense_intrinsics)
	print("\nTransform: ", realsense_to_ee_transform)
	# print("\nStatic Transform: ", realsense_to_static_transform)

	# assert False

	# print("Transform: ", realsense_to_ee_transform)
	# assert False

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	object_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics,realsense_to_ee_transform, detected_objects, object_queue))
	position_tracking = threading.Thread(target=position_loop, args=(object_queue,))
	vision.start()
	position_tracking.start()
	# vision.join()
	# position_tracking.join()