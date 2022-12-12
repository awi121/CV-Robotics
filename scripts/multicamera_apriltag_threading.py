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
from DetectObjectNew import DetectObjectNew

from perception import CameraIntrinsics
from utils import *

from autolab_core import RigidTransform, YamlConfig

# NOTE: change the static camera's intrinsic and transform files as well as the serial number to test each camera
# calibration quality seperately (compare to wrist end-effector as ground truth)

REALSENSE_INTRINSICS_EE = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

REALSENSE_INTRINSICS_STATIC = "calib/realsense_intrinsics_camera5.intr"
REALSENSE_STATIC_TF = "calib/realsense_camera5.tf"

def vision_loop(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# ------- Configure Static Camera Stream ------------
	pipeline1 = rs.pipeline()
	config1 = rs.config()
	config1.enable_device('151322066932')
	config1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	pipeline1.start(config1)
	aligned_stream1 = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud1 = rs.pointcloud()

	# ------- Configure Gripper Camera Stream ------------
	pipeline2 = rs.pipeline()
	config2 = rs.config()
	config2.enable_device('220222066259')
	config2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	pipeline2.start(config2)
	aligned_stream2 = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud2 = rs.pointcloud()

	# get a stream of images
	while True:
		current_pose = fa.get_pose()

		# -------- Static Get Frames and Detect Apriltags ----------
		frames1 = pipeline1.wait_for_frames()
		frames1 = aligned_stream1.process(frames1)
		color_frame1 = frames1.get_color_frame()
		depth_frame1 = frames1.get_depth_frame().as_depth_frame()

		points1 = point_cloud1.calculate(depth_frame1)
		verts1 = np.asanyarray(points1.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
		
		# Convert images to numpy arrays
		depth_image1 = np.asanyarray(depth_frame1.get_data())

		# skip empty frames
		if not np.any(depth_image1):
			print("no depth")
			# continue

		color_image1 = np.asanyarray(color_frame1.get_data())

		# TODO: color thresholding to turn color into more light values in black and white
			# pick the lightest value out of the RGB chanels???



		bw_image1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)

		# python wrapper AprilTag package
		detector1 = Detector(families="tag36h11",
			nthreads=1,
			quad_decimate=1.0,
			quad_sigma=0.0,
			refine_edges=1,
			decode_sharpening=0.25,
			debug=0)

		# camera parameters [fx, fy, cx, cy]
		cam_param1 = [realsense_intrinsics_static.fx, realsense_intrinsics_static.fy, realsense_intrinsics_static.cx, realsense_intrinsics_static.cy]
		detections1 = detector1.detect(bw_image1, estimate_tag_pose=True, camera_params=cam_param1, tag_size=0.03)

		# -------- Gripper Get Frames and Detect Apriltags ----------
		frames2 = pipeline2.wait_for_frames()
		frames2 = aligned_stream2.process(frames2)
		color_frame2 = frames2.get_color_frame()
		depth_frame2 = frames2.get_depth_frame().as_depth_frame()

		points2 = point_cloud2.calculate(depth_frame2)
		verts2 = np.asanyarray(points2.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
		
		# Convert images to numpy arrays
		depth_image2 = np.asanyarray(depth_frame2.get_data())

		# skip empty frames
		if not np.any(depth_image2):
			print("no depth")
			# continue

		color_image2 = np.asanyarray(color_frame2.get_data())
		bw_image2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)

		# python wrapper AprilTag package
		detector2= Detector(families="tag36h11",
			nthreads=1,
			quad_decimate=1.0,
			quad_sigma=0.0,
			refine_edges=1,
			decode_sharpening=0.25,
			debug=0)

		# camera parameters [fx, fy, cx, cy]
		cam_param2 = [realsense_intrinsics_ee.fx, realsense_intrinsics_ee.fy, realsense_intrinsics_ee.cx, realsense_intrinsics_ee.cy]
		detections2 = detector1.detect(bw_image2, estimate_tag_pose=True, camera_params=cam_param2, tag_size=0.03)

		# ------- Find Apriltags Both Cameras Detected -------
		# want a list with detection, and two booleans of static and gripper
		all_detections = [] # key = d.tag_id, value = full detectioni, static and gripper boolean
		both = []
		for d1 in detections1:
			obj1_id = d1.tag_id
			start_len = len(all_detections)
			for d2 in detections2:
				obj2_id = d2.tag_id

				# tag is visible in both cameras
				if obj1_id == obj2_id:
					all_detections.append((d1, d2))
					both.append(obj2_id)

			# if the tag isn't visible in both cameras the length of all_detections will not have changed
			if len(all_detections) - start_len == 0:
				all_detections.append((d1, None))

		for d2 in detections2:
			obj2_id = d2.tag_id

			# if d2 wasn't detected by both cameras it will not currently be in the all_detections list
			if both.count(obj2_id) == 0:
				all_detections.append((None, d2))

		verts = [verts1, verts2]

		# iterate all the detections in both cameras
		for d in all_detections:
			# booleans for which cameras the apriltag was detected by
			gripper = False 
			static = False 

			bounds = [] 
			translation = []

			if d[0] != None:
				obj_id = d[0].tag_id
				static = True

				# calculate bounds and verts in static frame
				(ptA, ptB, ptC, ptD) = d[0].corners
				ptB = (int(ptB[0]), int(ptB[1]))
				ptC = (int(ptC[0]), int(ptC[1]))
				ptD = (int(ptD[0]), int(ptD[1]))
				ptA = (int(ptA[0]), int(ptA[1]))

				# draw the bounding box of the AprilTag detection
				cv2.line(color_image1, ptA, ptB, (0, 255, 0), 2)
				cv2.line(color_image1, ptB, ptC, (0, 255, 0), 2)
				cv2.line(color_image1, ptC, ptD, (0, 255, 0), 2)
				cv2.line(color_image1, ptD, ptA, (0, 255, 0), 2)

				# draw the center (x, y)-coordinates of the AprilTag
				(cX1, cY1) = (int(d[0].center[0]), int(d[0].center[1]))
				cv2.circle(color_image1, (cX1, cY1), 5, (0, 0, 255), -1)

				# --------- added code to calculate AprilTag x,y,z position ------
				bounds.append(np.array([ptA, ptB, ptC, ptD]))
				translation_matrix = d[0].pose_t
				translation.append(np.array(translation_matrix).reshape(3))
				rotation_matrix = d[0].pose_R
				pose_error = d[0].pose_err

				# check if apriltag has been detected before
				if obj_id not in detected_objects:
					# add tag to the dictionary of detected objects
					tagFamily = d[0].tag_family.decode("utf-8")
					detected_objects.update({obj_id : DetectObjectNew(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, object_id=obj_id, object_class=tagFamily)})
				
			if d[1] != None:
				obj_id = d[1].tag_id
				gripper = True

				# calculate bounds and verts in static frame
				(ptA, ptB, ptC, ptD) = d[1].corners
				ptB = (int(ptB[0]), int(ptB[1]))
				ptC = (int(ptC[0]), int(ptC[1]))
				ptD = (int(ptD[0]), int(ptD[1]))
				ptA = (int(ptA[0]), int(ptA[1]))

				# draw the bounding box of the AprilTag detection
				cv2.line(color_image2, ptA, ptB, (0, 255, 0), 2)
				cv2.line(color_image2, ptB, ptC, (0, 255, 0), 2)
				cv2.line(color_image2, ptC, ptD, (0, 255, 0), 2)
				cv2.line(color_image2, ptD, ptA, (0, 255, 0), 2)

				# draw the center (x, y)-coordinates of the AprilTag
				(cX2, cY2) = (int(d[1].center[0]), int(d[1].center[1]))
				cv2.circle(color_image2, (cX2, cY2), 5, (0, 0, 255), -1)

				# --------- added code to calculate AprilTag x,y,z position ------
				bounds.append(np.array([ptA, ptB, ptC, ptD]))
				translation_matrix = d[1].pose_t
				translation.append(np.array(translation_matrix).reshape(3))
				rotation_matrix = d[1].pose_R
				pose_error = d[1].pose_err

				# check if apriltag has been detected before
				if obj_id not in detected_objects:
					# add tag to the dictionary of detected objects
					tagFamily = d[1].tag_family.decode("utf-8")
					detected_objects.update({obj_id : DetectObjectNew(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, object_id=obj_id, object_class=tagFamily)})
				
			# calculate position
			obj = detected_objects[obj_id]
			object_center_point = obj.get_position_apriltag(bounds, verts, current_pose, translation, static, gripper)

			string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
			cv2.putText(color_image1, string, (cX1 - 30, cY1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			# cv2.putText(color_image2, string, (cX2 - 30, cY2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# put updated dictionary in queue for other thread to access
		object_queue.put(detected_objects)

		# Show the images
		cv2.imshow("Image Static", color_image1)
		cv2.imshow("Image End Effctor", color_image2)
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
	parser.add_argument("--ee_intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_EE)
	parser.add_argument("--static_intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_STATIC)
	parser.add_argument("--ee_extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	parser.add_argument("--static_extrinsics_file_path", type=str, default=REALSENSE_STATIC_TF)
	args = parser.parse_args()
	
	# reset pose and joints
	fa = FrankaArm()
	# fa.reset_pose()
	# fa.reset_joints()

	# # move to center
	# pose = fa.get_pose()
	# # print("\nRobot Pose: ", pose)
	# pose.translation = np.array([0.6, 0, 0.5])
	# fa.goto_pose(pose)

	realsense_intrinsics_ee = CameraIntrinsics.load(args.ee_intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.ee_extrinsics_file_path)
	realsense_intrinsics_static = CameraIntrinsics.load(args.static_intrinsics_file_path)
	realsense_to_static_transform = RigidTransform.load(args.static_extrinsics_file_path)

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	object_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, detected_objects, object_queue))
	position_tracking = threading.Thread(target=position_loop, args=(object_queue,))
	vision.start()
	position_tracking.start()
	# vision.join()
	# position_tracking.join()