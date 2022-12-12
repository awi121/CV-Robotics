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
	fa.open_gripper()

	# move to middle scanning position
	pose = fa.get_pose()
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	color_image = get_realsense_rgb_image(cv_bridge)
	depth_image = get_realsense_depth_image(cv_bridge)

	object_image_position = np.array([200, 300])

	blur_image = cv2.GaussianBlur(color_image, (5,5),5)

	# convert image to HSV (hue-saturation-value) space
	hsv_frame = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

	# define the green color mask
	green_lower = np.array([25, 52, 72], np.uint8)
	green_upper = np.array([102, 255, 255], np.uint8)
	green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

	# Add kernal for dilation to further help to reduce noise
	kernal = np.ones((5,5), "uint8")

	# modify the masks with dilation
	green_mask = cv2.dilate(green_mask, kernal)

	# apply masks to original image
	res_green = cv2.bitwise_and(color_image, color_image, mask=green_mask)

	# create contours
	green_cnt, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# get robot current pose
	current_pose = fa.get_pose()

	for cnt in green_cnt:
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

			# draw the contour
			cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

	# detect edges
	green_edges = cv2.Canny(green_mask,100,100)

	# Show the images
	cv2.imshow("Image", color_image)
	# cv2.imshow("Green Masked", res_green)
	# cv2.imshow("Green Edges", green_edges)
	cv2.waitKey(1000)

	input("Press Enter to continue if block detected...")

	# move the robot
	intermediate_pose_z_height = 0.19

	object_center_pose = current_pose

	object_center_pose.translation = [
		object_center_point_in_world[0],
		object_center_point_in_world[1],
		object_center_point_in_world[2],
	]

	intermediate_robot_pose = object_center_pose.copy()
	intermediate_robot_pose.translation = [
		object_center_point_in_world[0],
		object_center_point_in_world[1],
		intermediate_pose_z_height,
	]

	# Move to intermediate robot pose
	fa.goto_pose(intermediate_robot_pose)
	fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

	# Close Gripper
	# fa.goto_gripper(0.045, grasp=True, force=10.0)
	fa.goto_gripper(0.028, grasp=True, force=10.0)

	# Move to intermediate robot pose
	fa.goto_pose(intermediate_robot_pose)

	# define place pose
	place_pose = fa.get_pose()
	object_z_height = object_center_point_in_world[2] + 0.025
	place_pose.translation = [0.45, 0, object_z_height,]

	place_intermediate_pose = place_pose.copy()
	place_intermediate_pose.translation = [
		0.45,
		0,
		intermediate_pose_z_height,
	]

	fa.goto_pose(place_intermediate_pose)
	fa.goto_pose(place_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

	# Open Gripper
	print("Opening Grippers")
	fa.open_gripper()

	fa.goto_pose(intermediate_robot_pose)
	fa.reset_pose()

