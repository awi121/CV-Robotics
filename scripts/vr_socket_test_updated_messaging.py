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
import apriltag
from pupil_apriltags import Detector

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

### SESSION NOTES ###
# comment out print in utils.get_object_center_point_in_world_realsense_3D_camera_point (131)
# comment out print in DetectObject.py (185, 196, 178)

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

def vision_loop(realsense_intrinsics, realsense_to_ee_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_device('220222066259')
	config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	# print("[INFO] start streaming...")
	pipeline.start(config)

	aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud = rs.pointcloud()

	# get a stream of images
	while True:
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
		# if not np.any(depth_image):
		# 	print("no depth")
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

		# loop over the detected AprilTags
		for d in detections:

			# check if apriltag has been detected before
			obj_id = d.tag_id
			# if detected_objects.has_key(obj_id) == False:
			if obj_id not in detected_objects:
				# print("add to dictionary")
				# add tag to the dictionary of detected objects
				tagFamily = d.tag_family.decode("utf-8")
				detected_objects[obj_id] = DetectObject(object_id=obj_id, object_class=tagFamily)

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
			object_center_point = obj.get_position_apriltag(bounds, verts, current_pose, translation_matrix)

			string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
			cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# put updated dictionary in queue for other thread to access
		object_queue.put(detected_objects)

		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)


def new_object_message(new_object_list, object_dict):
	message = ""
	for new_object in new_object_list:
		new_object = int(new_object)
		message += '_newItem\t' + object_dict[new_object]._return_type() \
		+ '\t' + str(new_object) + '\t' + object_dict[new_object]._return_size() \
		+ '\t' + object_dict[new_object]._return_color() + '\n'
	return message
 
def object_message(object_name, object_dict):
	pos = object_dict[object_name]._return_current_position()
	vel = object_dict[object_name]._return_current_velocity()
	rot = object_dict[object_name]._return_current_rotation()
	avel = object_dict[object_name]._return_current_ang_velocity()
	return str(object_name) + '\t' + str(-pos[1]) + ',' + str(pos[2]) + ',' + str(pos[0]-0.6) + '\t' \
	+ str(-vel[1]) + ',' + str(vel[2]) + ',' + str(vel[0]) + '\t' \
	+ str(rot[1]) + ',' + str(-rot[2]) + ',' + str(-rot[0]) + ',' + str(rot[3]) + '\t' \
	+ str(avel[1]) + ',' + str(-avel[2]) + ',' + str(-avel[0])

# def control(object_queue, ):


if __name__ == "__main__":
	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()

	fa = FrankaArm()
	fa.reset_joints()
	grip_wrapper = GripperWrapper(fa)

	pose = fa.get_pose()
	goal_rotation = pose.quaternion
	print("Robot Resting Pose: ", pose)

	print('start socket')
	#change IP
	sock = U.UdpComms(udpIP="172.26.40.95", sendIP = "172.26.2.162", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
	message_index = 0
	new_object_list = [] # list of all of the objects to render
	inventory_list = []

	i = 0
	dt = 0.02
	
	rate = rospy.Rate(1 / dt)
	pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
	T = 1000
	max_speed = 4 #m/s
	break_acc = 20 #m/s^2

	fa = FrankaArm()
	print('about to reset')
	fa.reset_joints()
	pose = fa.get_pose()

	fa.goto_gripper(0, grasp=True)

	# go to scanning position
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning for blocks

	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	# NEED: detected_objects from queue
	object_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics,realsense_to_ee_transform, detected_objects, object_queue))
	vision.start()
	
	fa.goto_pose(pose)
	initialize = True
	hand_pose = pose
	goal_position = pose.translation

	while True:
		previous_pose = hand_pose
		hand_pose = fa.get_pose()
		hand_position = hand_pose.translation
		hand_rot = hand_pose.quaternion
		finger_width = 0.04 #fa.get_gripper_width() # check this 
		message_index += 1
		
		
		queue_size = object_queue.qsize()
		while queue_size > 0:
			# print("Queue got backed up - removing....")
			detected_objects = object_queue.get()
			queue_size = object_queue.qsize()

		# detected_objects = object_queue.get()
		
		send_string = str(message_index) + '\n'
		# print('detected_objects', detected_objects)
		# print('keys:', detected_objects.keys(), 'type:', type(detected_objects.keys()[0]))
		for item in inventory_list:
			if not (int(item)) in detected_objects.keys():
				send_string += "_deleteItem" + '\t' + item + '\n'
		for item in detected_objects.keys():
			if not(str(item) in inventory_list) and not(str(item) in new_object_list):
				new_object_list.append(str(item))
		if len(new_object_list) != 0:
			send_string += new_object_message(new_object_list, detected_objects)
		for game_object in detected_objects:
			send_string += object_message(game_object, detected_objects) + '\n'
		send_string += '_hand\t' + str(-hand_position[1]) + ',' + str(hand_position[2]) + ',' + str(hand_position[0]-0.6) +'\t'\
			+ str(hand_rot[2]) + ',' + str(-hand_rot[3]) + ',' + str(-hand_rot[1]) + ',' + str(hand_rot[0]) + '\t' + str(finger_width)

		sock.SendData(send_string) # Send this string to other application
		data = sock.ReadReceivedData() # read data

		# print("Data: ", data)

		if data != None: # if NEW data has been received since last ReadReceivedData function call
			# print('send_string', send_string)
			# print()
			# print(data)
			# print('\n')
			inventory, unknown_objects, gripper_data = data.split('\n')
			inventory_list = inventory.split('\t')[1:]
			new_object_list = unknown_objects.split('\t')[1:]
			previous_goal_position = goal_position

			goal_position, goal_rotation, goal_width = gripper_data.split('\t')
			cur_pose = hand_pose
			cur_position = cur_pose.translation
			goal_position = np.array(goal_position[1:-1].split(', ')).astype(np.float)
			goal_position = np.array([goal_position[2] + 0.6, -goal_position[0], goal_position[1] + 0.02])
			goal_rotation = np.array(goal_rotation[1:-1].split(', ')).astype(np.float)
			goal_rotation = np.array([goal_rotation[3], -goal_rotation[2], goal_rotation[0], -goal_rotation[1]])
			goal_width = 2*float(goal_width)

			goal_rotation_mat = pose.rotation_from_quaternion(goal_rotation)#@np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
			pose.rotation = goal_rotation_mat
			# fa.goto_pose(pose)
			# clip magnitude of goal position to be within max speed bounds
			if initialize:
				time_diff = dt
			else:
				time_diff = timestamp - last_time
				print("Time Diff:", time_diff)
				last_time = timestamp
			speed = np.linalg.norm(goal_position - cur_position)/time_diff
			goal_speed = np.linalg.norm(goal_position - previous_goal_position)/time_diff
			goal_speed = np.clip(goal_speed, 0, max_speed)
			goal_direction = (goal_position - previous_goal_position)/np.linalg.norm(goal_position - previous_goal_position)
			hand_speed = np.linalg.norm(cur_position- previous_pose.translation)/time_diff
			print('hand_speed', hand_speed)
			direction = (goal_position - cur_position)/np.linalg.norm(goal_position - cur_position)
			if speed > max_speed:
				# print('Modifying goal_position from:', goal_position)
				goal_position = max_speed*time_diff*direction + cur_position
				speed = max_speed
				# print('To:', goal_position)

			projected_pose = (hand_speed+speed)*time_diff*direction + goal_position
			while np.any([projected_pose <= FC.WORKSPACE_WALLS[:, :3].min(axis=0), projected_pose >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
				print((hand_speed+speed)*time_diff)
				goal_position -= direction*0.001
				projected_pose = (hand_speed+speed)*time_diff*direction + goal_position
			pose.translation = goal_position #+ goal_direction*goal_speed*time_diff
			
			if initialize:                   
				# terminate active skills
				fa.goto_pose(pose, duration=T, dynamic=True, buffer_time=10,
					cartesian_impedances=[600.0, 600.0, 600.0, 10.0, 10.0, 10.0])
				initialize = False

				init_time = rospy.Time.now().to_time()
				timestamp = rospy.Time.now().to_time() - init_time
				last_time = timestamp

			else:
				start_time = time.perf_counter()
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

				end_time = time.perf_counter()
				# print('run time:', end_time - start_time)

				#If the gripper is near a virtual wall, stop the gripper and re-initialize 
				#the goto_pose skill. This will 'freeze' the gripper and keep inertia from 
				#carying it into a wall.
				# Check if the gripper is about to run into a virtual wall.  
				# print(time_diff)
				# projected_pose = (hand_speed*3*time_diff + 0.5*hand_speed**2/break_acc)*direction + cur_position
				# if np.any([projected_pose <= FC.WORKSPACE_WALLS[:, :3].min(axis=0), projected_pose >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
				# 	fa.stop_skill()
				# 	initialize = True

			# grip_wrapper.goto(goal_width, 0.15)
			i+=1
			rate.sleep()