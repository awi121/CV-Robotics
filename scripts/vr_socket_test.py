import pickle as pkl
import numpy as np

from DetectObject import DetectObject

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk, min_jerk_weight

import rospy
import UdpComms as U
import time
import threading
import queue

import argparse
import cv2
import math
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *
# import sys
# sys.path.insert(1, './camera_calibration/scripts')
# import utils

# NOTE: In order to turn this script into live-immitating the VR controller pose/gripper width, we would need
# a subscriber in here that loads in the trajectory step by step. Instead of iterating through pose_traj in a 
# for loop, would have a while loop for while getting commands from VR ros node (or whatever we call it)

# NOTE: It actually would be helpful to get more steps in the trajectory, as one of the issues is that the 
# gripper closing can lag behind because the step size between trajectory points is quite large (e.g. 0.8 to 
# 0.6 in one step), which since the gripper and pose are decoupled can cause the timings to mis-align

def vision():
	# scan for a block
	REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
	REALSENSE_EE_TF = "calib/realsense_ee.tf"
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# create image class
	detection = DetectObject(object_id=0, object_class="block")

	while True:
		current_pose = fa.get_pose()

		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)

		# meaningless right now - placeholder for updates to the class
		object_bounds = [0,0]

		object_center_point = detection.get_position_image(color_image, depth_image, object_bounds, current_pose)
		
		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
		print("\nBlock Position: ", object_center_point)



# def control():


if __name__ == "__main__":

	# Multithreading notes:
		# def task1(): define what needs to run on one thread
		# def task2(): define what needs to run on the other thread
		# t1 = threading.Thread(target=task1, name='t1')
		# t2 = threading.Thread(target=task2, name='t2')
		# t1.start()
		# t2.start()
		# t1.join()
		# t2.join()

	fa = FrankaArm()
	fa.reset_joints()

	pose = fa.get_pose()
	goal_rotation = pose.quaternion
	print("Robot Resting Pose: ", pose)

	print('start socket')
	#change IP
	sock = U.UdpComms(udpIP="172.26.40.95", sendIP = "172.26.90.96", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

	i = 0
	dt = 0.02
	rate = rospy.Rate(1 / dt)
	pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
	T = 100
	max_speed = 1 #m/s

	fa = FrankaArm()
	fa.reset_joints()
	pose = fa.get_pose()

	fa.goto_gripper(0, grasp=True)

	# go to scanning position
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)



	# ---- remove for multithreading -----
	# scan for a block
	REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
	REALSENSE_EE_TF = "calib/realsense_ee.tf"
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# create image class
	detection = DetectObject(object_id=0, object_class="block")

	current_pose = fa.get_pose()

	color_image = get_realsense_rgb_image(cv_bridge)
	depth_image = get_realsense_depth_image(cv_bridge)

	# meaningless right now - placeholder for updates to the class
	object_bounds = [0,0]

	object_center_point = detection.get_position_image(color_image, depth_image, object_bounds, current_pose)
	obj_string = "{:f},{:f},{:f}".format(-object_center_point[1], object_center_point[2], object_center_point[0])
	print("\nBlock Position: ", object_center_point)
	
	intialize = True

	while True:
		if obj_string != "":
			new_message = obj_string + '\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0,0'
			sock.SendData(new_message) # Send this string to other application

			# print("New Message: ", new_message)
		
		data = sock.ReadReceivedData() # read data

		# print("Data: ", data)

		if data != None: # if NEW data has been received since last ReadReceivedData function call
			print(data)
			goal_position, goal_width = data.split('\t')
			cur_pose = fa.get_pose()
			cur_position = cur_pose.translation
			goal_position = np.array(goal_position[1:-1].split(', ')).astype(np.float)
			goal_position = np.array([goal_position[2] + 0.6, -goal_position[0], goal_position[1] + 0.02])
			goal_width = float(goal_width)

			# clip magnitude of goal position to be within max speed bounds
			if not intialize:
				time_diff = timestamp - last_time
				last_time = timestamp
				print("Time Diff:", time_diff)
				if np.linalg.norm(goal_position - cur_position) > max_speed*time_diff:
					goal_position = max_speed*time_diff*(goal_position - cur_position)/np.linalg.norm(goal_position - cur_position) + cur_position

			pose.translation = goal_position
			# pose.quaternion = goal_rotation

			if intialize:
				# terminate active skills

				fa.goto_pose(pose)
				fa.goto_pose(pose, duration=T, dynamic=True, buffer_time=10,
					cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
				intialize = False

				init_time = rospy.Time.now().to_time()
				timestamp = rospy.Time.now().to_time() - init_time
				last_time = timestamp
			else:
				timestamp = rospy.Time.now().to_time() - init_time
				traj_gen_proto_msg = PosePositionSensorMessage(
					id=i, timestamp=timestamp,
					position=pose.translation, quaternion=pose.quaternion
				)
				ros_msg = make_sensor_group_msg(
					trajectory_generator_sensor_msg=sensor_proto2ros_msg(
						traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
					)

				rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
				pub.publish(ros_msg)
				rate.sleep()
			# fa.goto_pose(pose)

			i+=1
			#goal_width = 2*float(data)
			# fa.goto_gripper(goal_width, block=False, grasp=False, speed=0.1)
			# print("\nDesired Gripper Width: ", goal_width, "Current Gripper State: ", fa.get_gripper_width())
			# time.sleep(1)

			# SOLUTION BRAINSTORM:
				# Current Problems: Latent delay, when grasp=True there is an overshoot
				# and gripper oscillation (with speed fixed to 0.15 this is particularly bad)

				# try terminating grasp action before sending another goto_gripper command
				# have a check if object in view before grasp and only if so, then use grasp = True
				# have x + delta_t * v to scale the desired gripper width to account for latency 

			# min_gripper_delay = 0.2
			# gripper_update_rate = 1 + int(min_gripper_delay/dt)
			# if i % gripper_update_rate == 0:
			# 	difference = goal_width - fa.get_gripper_width()
			# 	speed = abs(difference/(dt*gripper_update_rate ))
			# 	# print("\nSpeed: ", speed)
			# 	if speed > 0.15:
			# 	    speed = 0.15
			# 	# speed = 0.15
			# 	if abs(difference) > 0.001:
			# 		if difference >= 0:
			# 			# print("opening")
			# 			fa.goto_gripper(goal_width, block=False, grasp=False, speed=speed)
			# 		else:
			# 			# print("closing")
			# 			fa.goto_gripper(goal_width, block=False, grasp=False, speed=speed)
			# 		print("i:  ", i, "   Desired Gripper Width: ", goal_width, "Current Gripper State: ", fa.get_gripper_width())
				# else:
					# print("do nothing :)")

			# rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
			# pub.publish(ros_msg)
			rate.sleep()


			current_pose = fa.get_pose()

			color_image = get_realsense_rgb_image(cv_bridge)
			depth_image = get_realsense_depth_image(cv_bridge)

			# meaningless right now - placeholder for updates to the class
			object_bounds = [0,0]

			object_center_point = detection.get_position_image(color_image, depth_image, object_bounds, current_pose)
			obj_string = "{:f},{:f},{:f}".format(-object_center_point[1], object_center_point[2], object_center_point[0])
			print("\nBlock Position: ", object_center_point)
