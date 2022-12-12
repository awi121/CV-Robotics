from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
import math
import random
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

if __name__ == "__main__":
	"""
	This is a script that will scan for the clay in the middle of the workspace,
	identify the outline and center point using classical CV methods, then
	randomly execute 10 grasp squish actions about the clay. This is a great
	script for testing gripper fingertips, etc.
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
	# fa.reset_pose()
	# fa.reset_joints()

	# loop for 10 random actions
	for i in range(500):
		# move to middle scanning position
		# pose = fa.get_pose()
		# pose.translation = np.array([0.6, 0, 0.5])
		# fa.goto_pose(pose)

		# begin scanning blocks based on colors
		cv_bridge = CvBridge()
		realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
		realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

		current_pose = fa.get_pose()

		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)
		object_image_position = np.array([200, 300])
		blur_image = cv2.GaussianBlur(color_image, (5,5),5)
		img = color_image

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

		# detect edges
		edges = cv2.Canny(gray_mask, 100,100)
		cv2.imshow("Edges", edges)

		# apply new contour mask to color image
		res_color = cv2.bitwise_and(color_image, color_image, mask = cont_mask)
		# cv2.imshow("Res Color", res_color)

		# draw/calculate the centers of objects
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 300:
				print("\n\nfinding center...")
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])

				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					realsense_intrinsics,
					realsense_to_ee_transform,
					current_pose)

				string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2])
				area_string = "Area: {:0.2f} [pixel]".format(area)

				# draw contours, COM and area on color image
				cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)
				cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.putText(color_image, area_string, (cX - 35, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

				# draw contours, COM and area on res image
				cv2.circle(res_color, (cX, cY), 7, (255,255,255), -1)
				cv2.putText(res_color, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.putText(res_color, area_string, (cX - 35, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.drawContours(res_color, [cnt], 0, (0,0,255), 2)

		# Show the images
		cv2.imshow("Image", color_image)
		cv2.imshow("Res Color", res_color)
		cv2.waitKey(500)

		# # Save the raw color image to images folder
		# filename = "scripts/Images/shape_test_" + str(i) + ".jpg"
		# cv2.imwrite(filename, img)

		# # NOTE: have a popup that has the user approve the location/grasp width (replace this with the GUI control)
		
		# # Move down and close gripper
		# random_x = random.randint(-2, 2)/100.0
		# width = random.randint(3, 6)/100.0
		# z_clipped = np.clip(object_center_point_in_world[2]/1.5, a_min=0.014, a_max=None)
		# ee_location = np.array([object_center_point_in_world[0]+random_x, object_center_point_in_world[1], z_clipped])
		
		# # print("\nLocation = ", ee_location)
		# # print("Gripper Width = ", width)
		# # input("Press ENTER to approve the above location...")

		# pose = fa.get_pose()
		# pose.translation = ee_location
		# fa.goto_pose(pose)
		# fa.goto_gripper(width)
		# fa.open_gripper()

	# pose = fa.get_pose()
	# pose.translation = np.array([0.6, 0, 0.5])
	# fa.goto_pose(pose)