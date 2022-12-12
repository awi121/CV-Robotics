from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
import math
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *

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
	fa.reset_pose()
	fa.reset_joints()

	# move to middle scanning position
	pose = fa.get_pose()
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# get a stream of images
	while True:
		# current_pose = fa.get_pose()

		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)

		object_image_position = np.array([200, 300])

		blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# convert image to HSV (hue-saturation-value) space
		hsv_frame = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

		# define the red color mask
		red_lower = np.array([136, 87, 111], np.uint8)
		red_upper = np.array([180, 255, 255], np.uint8)
		# red_lower = np.array([150, 90, 100], np.uint8)
		# red_upper = np.array([225, 255, 255], np.uint8)
		# red_lower = np.array([18, 40, 90], np.uint8)
		# red_upper = np.array([27, 255, 255], np.uint8)
		red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

		# define the yellow color mask
		yellow_lower = np.array([20, 150, 150], np.uint8)
		yellow_upper = np.array([30, 255, 255], np.uint8)
		yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

		# define the green color mask
		green_lower = np.array([25, 52, 72], np.uint8)
		green_upper = np.array([102, 255, 255], np.uint8)
		# green_lower = np.array([45, 70, 72], np.uint8)
		# green_upper = np.array([95, 255, 255], np.uint8)
		green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

		# define the blue color mask
		blue_lower = np.array([94, 80, 2], np.uint8)
		blue_upper = np.array([120, 255, 255], np.uint8)
		blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

		# Add kernal for dilation to further help to reduce noise
		kernal = np.ones((5,5), "uint8")

		# modify the masks with dilation
		red_mask = cv2.dilate(red_mask, kernal)
		yellow_mask = cv2.dilate(yellow_mask, kernal)
		green_mask = cv2.dilate(green_mask, kernal)
		blue_mask = cv2.dilate(blue_mask, kernal)

		# apply masks to original image
		res_red = cv2.bitwise_and(color_image, color_image, mask=red_mask)
		res_yellow = cv2.bitwise_and(color_image, color_image, mask=yellow_mask)
		res_green = cv2.bitwise_and(color_image, color_image, mask=green_mask)
		res_blue = cv2.bitwise_and(color_image, color_image, mask=blue_mask)

		# create contours
		red_cnt, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		yellow_cnt, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		green_cnt, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		blue_cnt, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# get robot current pose
		current_pose = fa.get_pose()

		# draw the contours onto the image
		for cnt in red_cnt:
			area = cv2.contourArea(cnt)
			if area > 300:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)


				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					realsense_intrinsics,
					realsense_to_ee_transform,
					current_pose)

				string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


				# # get the coordinates associated with the contour
				# contours = np.zeros((cnt.shape[0], 2), dtype=int)
				# for i in range(cnt.shape[0]):
				# 	contours[i] = [int(cnt[i][0][0]), int(cnt[i][0][1])]
				# cnt_points = verts[contours[:,1], contours[:,0]].reshape(-1,3)
				# z = np.median(cnt_points[:,2])
				# y = np.median(cnt_points[:,1])
				# x = np.median(cnt_points[:,0])
				# # print("\nRed Contour Points: (", x, ",", y, ",", z, ")")

				# # get the coordinates associated with the min and max
				# col_min = np.amin(cnt, axis=0)[0]
				# col_max = np.amax(cnt, axis=0)[0]
				# minmax_points = verts[col_min[1]:col_max[1], col_min[0]:col_max[0]].reshape(-1,3)
				# z_pos = np.median(minmax_points[:,2])
				# y_pos = np.median(minmax_points[:,1])
				# x_pos = np.median(minmax_points[:,0])
				# # print("Red MinMax Points: (", x_pos, ",", y_pos, ",", z_pos, ")")

				# # get the center of the block coordinates
				# yellow_point = verts[cY, cX].reshape(-1, 3)
				# # print("Red Center Point: ", yellow_point)

				# # print the x,y,z location on the screen
				# string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(x, y, z)
				# cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# draw the contour
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		for cnt in yellow_cnt:
			area = cv2.contourArea(cnt)
			if area > 300:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					realsense_intrinsics,
					realsense_to_ee_transform,
					current_pose)

				string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# # get the coordinates associated with the contour
				# contours = np.zeros((cnt.shape[0], 2), dtype=int)
				# for i in range(cnt.shape[0]):
				# 	contours[i] = [int(cnt[i][0][0]), int(cnt[i][0][1])]
				# cnt_points = verts[contours[:,1], contours[:,0]].reshape(-1,3)
				# z = np.median(cnt_points[:,2])
				# y = np.median(cnt_points[:,1])
				# x = np.median(cnt_points[:,0])
				# # print("\nYellow Contour Points: (", x, ",", y, ",", z, ")")

				# # get the coordinates associated with the min and max
				# col_min = np.amin(cnt, axis=0)[0]
				# col_max = np.amax(cnt, axis=0)[0]
				# minmax_points = verts[col_min[1]:col_max[1], col_min[0]:col_max[0]].reshape(-1,3)
				# z_pos = np.median(minmax_points[:,2])
				# y_pos = np.median(minmax_points[:,1])
				# x_pos = np.median(minmax_points[:,0])
				# # print("Yellow MinMax Points: (", x_pos, ",", y_pos, ",", z_pos, ")")

				# # get the center of the block coordinates
				# yellow_point = verts[cY, cX].reshape(-1, 3)
				# # print("Yellow Center Point: ", yellow_point)

				# # print the x,y,z location on the screen
				# string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(x, y, z)
				# cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# # estimate block pose using solvePnP
				# points2d = contours.astype(np.float32)
				# points3d = cnt_points

				# dist = np.zeros((4,1)) # assuming no lense distortion
				# ret, rvecs, tvecs = cv2.solvePnP(points3d, points2d, camera_matrix, dist)

				# draw the contour
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		for cnt in green_cnt:
			area = cv2.contourArea(cnt)
			if area > 1500:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					realsense_intrinsics,
					realsense_to_ee_transform,
					current_pose)

				string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# # get the coordinates associated with the contour
				# contours = np.zeros((cnt.shape[0], 2), dtype=int)
				# for i in range(cnt.shape[0]):
				# 	contours[i] = [int(cnt[i][0][0]), int(cnt[i][0][1])]
				# cnt_points = verts[contours[:,1], contours[:,0]].reshape(-1,3)
				# z = np.median(cnt_points[:,2])
				# y = np.median(cnt_points[:,1])
				# x = np.median(cnt_points[:,0])
				# # print("\nGreen Contour Points: (", x, ",", y, ",", z, ")")

				# # get the coordinates associated with the min and max
				# col_min = np.amin(cnt, axis=0)[0]
				# col_max = np.amax(cnt, axis=0)[0]
				# minmax_points = verts[col_min[1]:col_max[1], col_min[0]:col_max[0]].reshape(-1,3)
				# z_pos = np.median(minmax_points[:,2])
				# y_pos = np.median(minmax_points[:,1])
				# x_pos = np.median(minmax_points[:,0])
				# # print("Green MinMax Points: (", x_pos, ",", y_pos, ",", z_pos, ")")

				# # get the center of the block coordinates
				# yellow_point = verts[cY, cX].reshape(-1, 3)
				# # print("Green Center Point: ", yellow_point)

				# # print the x,y,z location on the screen
				# string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(x, y, z)
				# cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# draw the contour
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		for cnt in blue_cnt:
			area = cv2.contourArea(cnt)
			if area > 300:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					realsense_intrinsics,
					realsense_to_ee_transform,
					current_pose)

				string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# # get the coordinates associated with the contour
				# contours = np.zeros((cnt.shape[0], 2), dtype=int)
				# for i in range(cnt.shape[0]):
				# 	contours[i] = [int(cnt[i][0][0]), int(cnt[i][0][1])]
				# cnt_points = verts[contours[:,1], contours[:,0]].reshape(-1,3)
				# z = np.median(cnt_points[:,2])
				# y = np.median(cnt_points[:,1])
				# x = np.median(cnt_points[:,0])
				# # print("\nBlue Contour Points: (", x, ",", y, ",", z, ")")

				# # get the coordinates associated with the min and max
				# col_min = np.amin(cnt, axis=0)[0]
				# col_max = np.amax(cnt, axis=0)[0]
				# minmax_points = verts[col_min[1]:col_max[1], col_min[0]:col_max[0]].reshape(-1,3)
				# z_pos = np.median(minmax_points[:,2])
				# y_pos = np.median(minmax_points[:,1])
				# x_pos = np.median(minmax_points[:,0])
				# # print("Blue MinMax Points: (", x_pos, ",", y_pos, ",", z_pos, ")")

				# # get the center of the block coordinates
				# yellow_point = verts[cY, cX].reshape(-1, 3)
				# # print("Blue Center Point: ", yellow_point)

				# # print the x,y,z location on the screen
				# string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(x, y, z)
				# cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

				# draw the contour
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		# detect edges
		red_edges = cv2.Canny(red_mask,100,100)
		yellow_edges = cv2.Canny(yellow_mask,100,100)
		green_edges = cv2.Canny(green_mask,100,100)
		blue_edges = cv2.Canny(blue_mask,100,100)

		# Show the masked images
		cv2.imshow("Image", color_image)
		# cv2.imshow("Red Masked", res_red)
		# cv2.imshow("Yellow Masked", res_yellow)
		cv2.imshow("Green Masked", res_green)
		# cv2.imshow("Blue Masked", res_blue)

		# show the edges
		# cv2.imshow("Red Edges", red_edges)
		# cv2.imshow("Yellow Edges", yellow_edges)
		cv2.imshow("Green Edges", green_edges)
		# cv2.imshow("Blue Edges", blue_edges)

		cv2.waitKey(1)


