import numpy as np
import time

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import convert_array_to_rigid_transform

import rospy

class GripperWrapper:
	def __init__(self, fa, close_tolerance = 0.001):
		self.fa = fa
		self.closed = False
		self.last_width = 0.08
		self.close_tolerance = close_tolerance

	def goto(self, width, speed):
		cur_width = fa.get_gripper_width()
		if cur_width - self.last_width > self.close_tolerance and self.closed:
			squeezing = True
		else:
			squeezing = False
		diff = width - fa.get_gripper_width()
		if abs(diff) > 0.001:
			if diff > 0 and squeezing:
				print('stop_grip')
				fa.stop_gripper()
			if diff < 0:
				self.closed = True
			else:
				self.closed = False
			fa.goto_gripper(width, block=False, grasp=False, speed=speed)
			self.last_width = width

# def gripper_wrapper(width, fa, delay):
# 	difference = width - fa.get_gripper_width()
# 	speed = abs(difference/delay)
# 	# print("\nSpeed: ", speed)
# 	if speed > 0.15:
# 	    speed = 0.15
# 	speed = 0.15
# 	if abs(difference) > 0.001:
# 		if difference > 0:
# 			# print("opening")
# 			fa.goto_gripper(width, block=False, grasp=False, speed=speed, force = 10)
# 		else:
# 			# print("closing")
# 			fa.goto_gripper(width, block=False, grasp=False, speed=speed, force = 10)
# 		print("Desired Gripper Width: ", width, "Current Gripper State: ", fa.get_gripper_width())
# 	else:
# 		print("do nothing :)")



if __name__ == "__main__":
	fa = FrankaArm()
	# # fa.reset_joints()
	# delay = 1
	# grip = GripperWrapper(fa)

	# gripper_widths = [0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
	# 				0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0, 0.08, 0.08, 0, 0, 0.08, 0]
	# # gripper_widths = [0, 0.08, 0.08, 0.08, 0, 0.01, 0.01, 0.01,0.01,  0.08, 0.08, 0, 0, 0.08, 0]

	# for width in gripper_widths:
	# 	grip.goto(width, 0.15)
	# 	time.sleep(delay)
	# 	print("Desired Gripper Width: ", width, "Current Gripper State: ", fa.get_gripper_width())

	# reset pose to center
	fa.reset_joints()
	pose = fa.get_pose()
	# print("\nRobot Pose: ", pose)
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# test:
	# fa.open_gripper()
	# time.sleep(5)
	# fa.goto_gripper(0.05, force=100.0)
	# time.sleep(5)

	# move with bottle
	pose = fa.get_pose()
	# print("\nRobot Pose: ", pose)
	pose.translation = np.array([0.7, 0, 0.25])
	fa.goto_pose(pose)

	# rotate joint to simulate pouring
	joints = fa.get_joints()

	print("Joints: ", joints)

	# joint 6: rotate wrist, joint 5: angle of gripper 
	# joints[6] += np.deg2rad(10)
	# fa.goto_joints(joints, ignore_virtual_walls=True)

	# joint 5
	joints[5] += np.deg2rad(-25)
	fa.goto_joints(joints, ignore_virtual_walls=True)



	# SOLUTION BRAINSTORM:
	# 	Current Problems: Latent delay, when grasp=True there is an overshoot
	# 	and gripper oscillation (with speed fixed to 0.15 this is particularly bad)

	# 	try terminating grasp action before sending another goto_gripper command
	# 	have a check if object in view before grasp and only if so, then use grasp = True
	# 	have x + delta_t * v to scale the desired gripper width to account for latency 
