from frankapy import FrankaArm
from DetectObject import DetectObject
import numpy as np
import argparse
import cv2
import math
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
import pyrealsense2 as rs

from perception import CameraIntrinsics
from utils import *

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

if __name__ == "__main__":
	"""
	This script uses classical CV methods to find the outlines of the objects in
	the scene as well as their x,y,z coordinates in the robot frame.
	"""

	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()
	
	# reset pose and joints
	fa = FrankaArm()
	fa.reset_pose()
	fa.reset_joints()

	# move to middle scanning position
	pose = fa.get_pose()
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning blocks based on colors
	# cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# create image class
	detection = DetectObject(object_id=0, object_class="block")


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

	# get a stream of images
	# while True:
	for i in range(15):
		pose = fa.get_pose()
		pose.translation = np.array([0.6, 0, 0.45-i*0.02])
		fa.goto_pose(pose)

		# beginning of loop
		current_pose = fa.get_pose()

		# # -------- Code for original DetectObject code --------
		# color_image = get_realsense_rgb_image(cv_bridge)
		# depth_image = get_realsense_depth_image(cv_bridge)

		# # with np.printoptions(threshold=np.inf):
		# # 	print("Depth Image: ", depth_image)

		# print("Depth Image Shape: ", depth_image.shape)
		# print("Other Depth Shape: ", depth_image[0].shape)

		# object_image_position = np.array([200, 300])

		# # meaningless right now - placeholder for updates to the class
		# object_bounds = [0,0]

		# object_center_point = detection.get_position_image(color_image, depth_image, object_bounds, current_pose)
		# print("Avg Center Point: ", object_center_point)



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

		# image class code begins
		blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# adaptive thresholding on greyscale image
		gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		kernal = np.ones((5,5), "uint8")
		gray_mask = cv2.dilate(thresh, kernal)

		# create contours
		contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cont_mask = np.zeros(gray_mask.shape, dtype='uint8')
		cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
		cv2.imshow("Contours", cont_mask)

		# threshold to only list contours above a certain area - after this, should only have 1!!!!
		# print("\nCountour: ", contours)

		# print("\nN Contours: ", len(contours))

		# draw/calculate the centers of objects
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 800:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				# width = int(np.sqrt(area)/8)

				# pixel_pairs = []
				# cx_start = cX - width
				# cy_start = cY - width
				# for i in range(2*width):
				# 	x = cx_start + i
				# 	for j in range(2*width):
				# 		y = cy_start + j
				# 		pixel_pairs.append([x,y])

				# object_center_point_in_world, variance = get_object_center_point_in_world_realsense_robust(
				# 	cX,
				# 	cY,
				# 	pixel_pairs,
				# 	depth_image,
				# 	realsense_intrinsics,
				# 	realsense_to_ee_transform,
				# 	current_pose)

				# # if variance is too high, then ignore z position update
				# if variance > 1e-4:
				# 	print("high variance....ignore z update")
				# 	object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point[2]])
				# else:
				# 	object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])

				

				# re-format the cnt to be [[y1,x1], [y2,x2], ...]
				bounds = np.array(cnt).reshape(len(cnt),2)

				minx = np.amin(bounds[:,0], axis=0)
				maxx = np.amax(bounds[:,0], axis=0)
				miny = np.amin(bounds[:,1], axis=0)
				maxy = np.amax(bounds[:,1], axis=0)
				
				obj_points = verts[miny:maxy, minx:maxx].reshape(-1,3)

				zs = obj_points[:,2]
				z = np.median(zs)
				xs = obj_points[:,0]
				ys = obj_points[:,1]
				ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background

				x_pos = np.median(xs)
				y_pos = np.median(ys)
				z_pos = z

				variance = np.var(zs) # NOTE: variance > 0.15, then z incorrect????

				median_point = np.array([x_pos, y_pos, z_pos])
				object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, realsense_intrinsics, realsense_to_ee_transform, current_pose)

				if variance > 0.015:
					print("High variance in z, ignore estimate...")
					object_center_point = np.array([object_median_point[0], object_median_point[1], object_center_point[2]])
				else:
					object_center_point = object_median_point

				print("Object Median Point: ", object_median_point)

				center = verts[cY,cX]
				object_center = get_object_center_point_in_world_realsense_3D_camera_point(center, realsense_intrinsics, realsense_to_ee_transform, current_pose)

				print("Center: ", object_center)
				print("Variance: ", variance)


				# assert False

				# image class code ends
				string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
				area_string = "Area: {:0.2f} [pixel]".format(area)

				# draw contours, COM and area on color image
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				# cv2.putText(color_image, area_string, (cX - 35, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)


		# # ------- Code for original vision stuff without the separate class -----
		# # image class code begins
		# blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# # adaptive thresholding on greyscale image
		# gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		# kernal = np.ones((5,5), "uint8")
		# gray_mask = cv2.dilate(thresh, kernal)

		# # create contours
		# contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cont_mask = np.zeros(gray_mask.shape, dtype='uint8')
		# cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
		# cv2.imshow("Contours", cont_mask)

		# # threshold to only list contours above a certain area - after this, should only have 1!!!!
		# # print("\nCountour: ", contours)

		# # print("\nN Contours: ", len(contours))

		# # draw/calculate the centers of objects
		# for cnt in contours:
		# 	area = cv2.contourArea(cnt)
		# 	if area > 800:
		# 		# compute the center of the contour
		# 		M = cv2.moments(cnt)
		# 		cX = int(M["m10"] / M["m00"])
		# 		cY = int(M["m01"] / M["m00"])
		# 		width = int(np.sqrt(area)/8)

		# 		pixel_pairs = []
		# 		cx_start = cX - width
		# 		cy_start = cY - width
		# 		for i in range(2*width):
		# 			x = cx_start + i
		# 			for j in range(2*width):
		# 				y = cy_start + j
		# 				pixel_pairs.append([x,y])

		# 		object_center_point_in_world, variance = get_object_center_point_in_world_realsense_robust(
		# 			cX,
		# 			cY,
		# 			pixel_pairs,
		# 			depth_image,
		# 			realsense_intrinsics,
		# 			realsense_to_ee_transform,
		# 			current_pose)

		# 		# if variance is too high, then ignore z position update
		# 		if variance > 1e-4:
		# 			print("high variance....ignore z update")
		# 			object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point[2]])
		# 		else:
		# 			object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])


		# 		# image class code ends
		# 		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
		# 		area_string = "Area: {:0.2f} [pixel]".format(area)

		# 		# draw contours, COM and area on color image
		# 		cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)
		# 		cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# 		cv2.putText(color_image, area_string, (cX - 35, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# 		cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		# # Show the images
		# cv2.imshow("Image", color_image)
		# cv2.waitKey(1)