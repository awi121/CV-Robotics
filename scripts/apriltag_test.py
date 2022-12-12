from frankapy import FrankaArm
import numpy as np
import argparse
import apriltag
from pupil_apriltags import Detector
import cv2
import math
import random
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
import pyrealsense2 as rs
from DetectObject import DetectObject

from perception import CameraIntrinsics
from utils import *

from autolab_core import RigidTransform, YamlConfig
# from perception_utils.apriltags import AprilTagDetector
# from perception_utils.realsense import get_first_realsense_sensor

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

if __name__ == "__main__":
	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()
	
	# reset pose and joints
	fa = FrankaArm()
	# fa.reset_pose()
	# fa.reset_joints()

	# # move to center
	# pose = fa.get_pose()
	# print("\nRobot Pose: ", pose)
	# pose.translation = np.array([0.6, 0, 0.5])
	# fa.goto_pose(pose)

	# cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	print("[INFO] start streaming...")
	pipeline.start(config)

	aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud = rs.pointcloud()

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}



	# get a stream of images
	# while True:
	for i in range(50):
		# color_image = get_realsense_rgb_image(cv_bridge)
		# depth_image = get_realsense_depth_image(cv_bridge)



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

		print("\n[INFO] found a valid depth frame")
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
		print("\nNumber of AprilTags: ", len(detections))

		# loop over the detected AprilTags
		for d in detections:

			# check if apriltag has been detected before
			obj_id = d.tag_id
			# if detected_objects.has_key(obj_id) == False:
			if obj_id not in detected_objects:
				print("add to dictionary")
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

			# # draw the tag family on the image
			# tagFamily = d.tag_family.decode("utf-8")
			# # cv2.putText(color_image, tagFamily, (ptA[0], ptA[1] - 15),
			# # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# # print("[INFO] tag family: {}".format(tagFamily))


			# --------- added code to calculate AprilTag x,y,z position ------
			bounds = np.array([ptA, ptB, ptC, ptD])
			# obj = DetectObject(object_id=0, object_class=tagFamily)
			obj = detected_objects[obj_id]
			object_center_point = obj.get_position_apriltag(bounds, verts, current_pose)




			# # -------- original working code before abstracted to class ------
			# minx = np.amin(bounds[:,0], axis=0)
			# maxx = np.amax(bounds[:,0], axis=0)
			# miny = np.amin(bounds[:,1], axis=0)
			# maxy = np.amax(bounds[:,1], axis=0)
			
			# obj_points = verts[miny:maxy, minx:maxx].reshape(-1,3)

			# zs = obj_points[:,2]
			# z = np.median(zs)
			# xs = obj_points[:,0]
			# ys = obj_points[:,1]
			# ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background

			# x_pos = np.median(xs)
			# y_pos = np.median(ys)
			# z_pos = z

			# variance = np.var(zs) # NOTE: variance > 0.15, then z incorrect????

			# median_point = np.array([x_pos, y_pos, z_pos])
			# object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, realsense_intrinsics, realsense_to_ee_transform, current_pose)

			# if variance > 0.015:
			# 	print("High variance in z, ignore estimate...")
			# 	object_center_point = np.array([object_median_point[0], object_median_point[1], object_center_point[2]])
			# else:
			# 	object_center_point = object_median_point

			# print("Object Median Point: ", object_median_point)

			# center = verts[cY,cX]
			# object_center = get_object_center_point_in_world_realsense_3D_camera_point(center, realsense_intrinsics, realsense_to_ee_transform, current_pose)

			# print("Center: ", object_center)
			# print("Variance: ", variance)




			string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
			cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)








			# # ---------- POSE ESTIMATION CODE --------
			# print("Rotation Pose: ", d.pose_R)
			# print("Translation Pose: ", d.pose_t)
			# print("Pose Error: ", d.pose_err)

			# visualize orientation
			# imgpts, jac = cv2.projectPoints(d.corners, d.pose_R, d.pose_t, camera_mat, dist_mat)









			# TODO:
				# 1) convert rotation matrix w.r.t. camera frame to quaternion
				# 2) translate quaternion from camera frame to franka_tool frame
				# 3) given franka_tool to world quaternion translate the object orientation to world frame
				# 4) convert quaternion to axis-angle format for interpretability
				# 5) visualize the orientation in the image livestream


		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)

	print("\nDetected Objects: ", detected_objects)
	print("Element: ", [detected_objects[6].object_id, detected_objects[6].object_class, detected_objects[6].object_center_point])
	print("Final COM: ", detected_objects[6].return_current_position())
