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
	# fa.reset_pose()
	# fa.reset_joints()

	# # move to middle scanning position
	# pose = fa.get_pose()
	# pose.translation = np.array([0.6, 0, 0.5])
	# fa.goto_pose(pose)

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# get a stream of images
	while True:
		current_pose = fa.get_pose()

		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)

		object_image_position = np.array([200, 300])

		blur_image = cv2.GaussianBlur(color_image, (5,5),5)


		# # Adaptive Color Thresholding
		# (B, G, R) = cv2.split(blur_image)
		# # cv2.imshow("Blue", B)
		# cv2.imshow("Green", G)
		# # cv2.imshow("Red", R)

		# # green_channel = blur_image[:,:,1]

		# # cv2.imshow("Green Channel", green_channel)

		# # # thresholded = cv2.adaptiveThreshold(green_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1, 2)
		# # thresholded = cv2.adaptiveThreshold(green_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

		# # cv2.imshow("thresholded", thresholded)


		# # # simple thresholding
		# # (T, threshold) = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY)
		# # threshwithblur = cv2.medianBlur(threshold, 15,0)
		# # cv2.imshow("Simple Thresh", threshwithblur)

		# # adaptive thresholding test
		# gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		# blurred = cv2.GaussianBlur(gray, (7,7), 0)
		# (T, threshInv) = cv2.threshold(B, 230, 255, cv2.THRESH_BINARY)
		# cv2.imshow("simple thresh", threshInv)

		# (T, threshOts) = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		# cv2.imshow("otsu thresh", threshOts)

		# thresh = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
		# cv2.imshow("adapt thresh", thresh)

		# gaus_thresh = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
		# cv2.imshow("gauss thresh", gaus_thresh)




		# NOTE IDEA: have the original method I've developed, but perform adaptive gaussian thresholding to get rid
		# of some of the noise????



		# Static Color Thresholding:

		# # convert image to HSV (hue-saturation-value) space
		# hsv_frame = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

		# # define the green color mask
		# green_lower = np.array([25, 52, 72], np.uint8)
		# green_upper = np.array([102, 255, 255], np.uint8)
		# green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

		# test out color channel
		# (B, G, R) = cv2.split(blur_image)
		# cv2.imshow("Green", G)
		# cv2.imshow("Blue", B)
		# cv2.imshow("Red", R)

		# looking at the different channels, green block has a high green value and low red value, whereas yellow has high green and red


		# try adaptive thresholding on greyscale image
		gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		# try increasing the contrast
		# cv2.imshow("Gray", gray)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		# ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
		# create a mask with adaptive thresholding

		# then separate into different colors???

		# cv2.imshow("Gray Threshold", thresh)
		# cv2.imshow("Green Threshold", cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 6))
		# cv2.imshow("Blue Threshold", cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 6))
		# cv2.imshow("Red Threshold", cv2.adaptiveThreshold(R, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 6))
		
		kernal = np.ones((5,5), "uint8")
		gray_mask = cv2.dilate(thresh, kernal)
		# cv2.imshow("Gray Mask", gray_mask)
		# apply mask to original image
		# res_gray = cv2.bitwise_and(gray, gray, mask = gray_mask)

		# create contours
		contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cont_mask = np.zeros(gray_mask.shape, dtype='uint8')
		cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
		cv2.imshow("Contours", cont_mask)

		# apply new contour mask to color image
		res_color = cv2.bitwise_and(color_image, color_image, mask = cont_mask)
		cv2.imshow("Res Color", res_color)
		# NOTE: print the COM on the Res Color image

		# # apply color filtering to the res_color
		# green = np.array([120, 255, 255], np.uint8)
		# green_lower = np.array([green[0]-10, 55, 55], np.uint8)
		# green_upper = np.array([green[0]+10, 255, 255], np.uint8)
		# # green_lower = np.array([25, 52, 72], np.uint8)
		# # green_upper = np.array([102, 255, 255], np.uint8)

		# yellow = np.array([60, 255, 255], np.uint8)
		# yellow_lower = np.array([yellow[0]-10, 100, 100], np.uint8)
		# yellow_upper = np.array([yellow[0]+10, 255, 255], np.uint8)

		# red = np.array([0, 100, 100], np.uint8)
		# red_lower = np.array([red[0], 70, 70], np.uint8)
		# red_upper = np.array([red[0]+10, 255, 255], np.uint8)

		# blue = np.array([240, 100, 100], np.uint8)
		# blue_lower = np.array([blue[0]-10, 50, 50], np.uint8)
		# blue_upper = np.array([blue[0]+10, 255, 255], np.uint8)

		# hsv_frame = cv2.cvtColor(res_color, cv2.COLOR_BGR2HSV)
		# green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
		# yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
		# red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
		# blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

		# # modify the masks with dilation
		# green_mask = cv2.dilate(green_mask, kernal)
		# yellow_mask = cv2.dilate(yellow_mask, kernal)
		# red_mask = cv2.dilate(red_mask, kernal)
		# blue_mask = cv2.dilate(blue_mask, kernal)

		# # apply masks to original image
		# res_green = cv2.bitwise_and(color_image, color_image, mask=green_mask)
		# res_yellow = cv2.bitwise_and(color_image, color_image, mask=yellow_mask)
		# res_red = cv2.bitwise_and(color_image, color_image, mask=red_mask)
		# res_blue = cv2.bitwise_and(color_image, color_image, mask=blue_mask)

		# cv2.imshow("Res Green", res_green)
		# cv2.imshow("Res Yellow", res_yellow)
		# cv2.imshow("Res Red", res_red)
		# cv2.imshow("Res Blue", res_blue)



		# convert to LAB color space
		lab_image = cv2.cvtColor(res_color, cv2.COLOR_BGR2LAB)
		a_channel = lab_image[:,:,1]	# spectrum from green to red
		b_channel = lab_image[:,:,2]	# spectrum from yellow to blue
		print("Min B:", np.min(b_channel))
		print("Max B: ", np.max(b_channel))
		
		green_thresh = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		green_masked = cv2.bitwise_and(res_color, res_color, mask = green_thresh)
		cv2.imshow("Green Masked", green_masked)

		red_thresh = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		red_masked = cv2.bitwise_and(res_color, res_color, mask = red_thresh)
		cv2.imshow("Red Masked", red_masked)

		# NOTE: THE BLUE THRESHOLDING ISN'T WORKING VERY WELL!!! BUT ALL OTHERS SEEM DECENT
		blue_thresh = cv2.threshold(b_channel, 115, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		blue_masked = cv2.bitwise_and(res_color, res_color, mask = blue_thresh)
		cv2.imshow("Blue Masked", blue_masked)

		yellow_thresh = cv2.threshold(b_channel, 165, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		yellow_masked = cv2.bitwise_and(res_color, res_color, mask = yellow_thresh)
		cv2.imshow("Yellow Masked", yellow_masked)

		# green_cont, hierarchy = cv2.findContours(green_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



		# for cnt in green_cont:
		# 	area = cv2.contourArea(cnt)
		# 	if area > 300:
		# 		# compute the center of the contour
		# 		M = cv2.moments(cnt)
		# 		cX = int(M["m10"] / M["m00"])
		# 		cY = int(M["m01"] / M["m00"])
		# 		cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

		# 		object_center_point_in_world = get_object_center_point_in_world_realsense(
		# 			cX,
		# 			cY,
		# 			depth_image,
		# 			realsense_intrinsics,
		# 			realsense_to_ee_transform,
		# 			current_pose)

		# 		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
		# 		cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# 		# draw the contour
		# 		cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)







		# # get edges
		# gray_edges = cv2.Canny(gray_mask,100,100)
		# cv2.imshow("Gray Edges", gray_edges)

		# I need a function to delete all contours enclosed by a contour 

		# for cnt in contours:
		# 	area = cv2.contourArea(cnt)
		# 	if area > 300:
		# 		# compute the center of the contour
		# 		M = cv2.moments(cnt)
		# 		cX = int(M["m10"] / M["m00"])
		# 		cY = int(M["m01"] / M["m00"])
		# 		cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

		# 		object_center_point_in_world = get_object_center_point_in_world_realsense(
		# 			cX,
		# 			cY,
		# 			depth_image,
		# 			realsense_intrinsics,
		# 			realsense_to_ee_transform,
		# 			current_pose)

		# 		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
		# 		cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# 		# draw the contour
		# 		cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		
		# # Add kernal for dilation to further help to reduce noise
		# kernal = np.ones((5,5), "uint8")

		# # modify the masks with dilation
		# green_mask = cv2.dilate(green_mask, kernal)

		# # apply masks to original image
		# res_green = cv2.bitwise_and(color_image, color_image, mask=green_mask)

		# # create contours
		# green_cnt, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# # get robot current pose
		# current_pose = fa.get_pose()

		# # for cnt in green_cnt:
		# # 	area = cv2.contourArea(cnt)
		# # 	if area > 300:
		# # 		# compute the center of the contour
		# # 		M = cv2.moments(cnt)
		# # 		cX = int(M["m10"] / M["m00"])
		# # 		cY = int(M["m01"] / M["m00"])
		# # 		cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)

		# # 		object_center_point_in_world = get_object_center_point_in_world_realsense(
		# # 			cX,
		# # 			cY,
		# # 			depth_image,
		# # 			realsense_intrinsics,
		# # 			realsense_to_ee_transform,
		# # 			current_pose)

		# # 		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
		# # 		cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# # 		# draw the contour
		# # 		cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		# # detect edges
		# # green_edges = cv2.Canny(green_mask,100,100)

		# Show the images
		cv2.imshow("Image", color_image)
		# cv2.imshow("Green Masked", res_green)
		# cv2.imshow("Green Edges", green_edges)
		cv2.waitKey(1)
