import pickle as pkl
import numpy as np

from DetectObjectNew import DetectObjectNew

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

REALSENSE_INTRINSICS_EE = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

REALSENSE_INTRINSICS_STATIC = "calib/realsense_static_intrinsics.intr"
REALSENSE_STATIC_TF = "calib/realsense_static.tf"

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

def vision_loop(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# ------- Configure Static Camera Stream ------------
	pipeline1 = rs.pipeline()
	config1 = rs.config()
	config1.enable_device('151322061880')
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
		detections1 = detector1.detect(bw_image1, estimate_tag_pose=True, camera_params=cam_param1, tag_size=0.022)

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
		detections2 = detector1.detect(bw_image2, estimate_tag_pose=True, camera_params=cam_param2, tag_size=0.022)

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
	parser.add_argument("--ee_intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_EE)
	parser.add_argument("--static_intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_STATIC)
	parser.add_argument("--ee_extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	parser.add_argument("--static_extrinsics_file_path", type=str, default=REALSENSE_STATIC_TF)
	args = parser.parse_args()

	fa = FrankaArm()
	fa.reset_joints()
	grip_wrapper = GripperWrapper(fa)

	pose = fa.get_pose()
	goal_rotation = pose.quaternion
	print("Robot Resting Pose: ", pose)

	print('start socket')
	#change IP
	sock = U.UdpComms(udpIP="172.26.40.95", sendIP = "172.26.81.222", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
	message_index = 0
	new_object_list = [] # list of all of the objects to render
	inventory_list = []

	i = 0
	dt = 0.02
	
	rate = rospy.Rate(1 / dt)
	pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
	T = 1000
	max_speed = 1 #m/s

	fa = FrankaArm()
	print('about to reset')
	fa.reset_joints()
	pose = fa.get_pose()

	fa.goto_gripper(0, grasp=True)

	# go to scanning position
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	realsense_intrinsics_ee = CameraIntrinsics.load(args.ee_intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.ee_extrinsics_file_path)
	realsense_intrinsics_static = CameraIntrinsics.load(args.static_intrinsics_file_path)
	realsense_to_static_transform = RigidTransform.load(args.static_extrinsics_file_path)

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	object_queue = queue.Queue()

	# begin scanning for blocks
	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, detected_objects, object_queue))
	vision.start()
	
	
	initialize = True
	while True:
		hand_pose = fa.get_pose()
		hand_position = hand_pose.translation
		hand_rot = hand_pose.quaternion
		finger_width = fa.get_gripper_width() # check this 
		message_index += 1

		queue_size = object_queue.qsize()
		while queue_size > 0:
			print("Queue got backed up - removing....")
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
		# new_message = obj_string + '\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0,0'
		sock.SendData(send_string) # Send this string to other application
		# print(send_string)
		# print("New Message: ", new_message)
	
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

			goal_position, goal_rotation, goal_width = gripper_data.split('\t')
			cur_pose = fa.get_pose()
			cur_position = cur_pose.translation
			goal_position = np.array(goal_position[1:-1].split(', ')).astype(np.float)
			goal_position = np.array([goal_position[2] + 0.6, -goal_position[0], goal_position[1] + 0.02])
			goal_rotation = np.array(goal_rotation[1:-1].split(', ')).astype(np.float)
			goal_rotation = np.array([goal_rotation[3], -goal_rotation[2], goal_rotation[0], -goal_rotation[1]])
			goal_width = 2*float(goal_width)

			pose.translation = goal_position
			print(goal_position[2])
			goal_rotation_mat = pose.rotation_from_quaternion(goal_rotation)#@np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
			pose.rotation = goal_rotation_mat
			# fa.goto_pose(pose)
			# clip magnitude of goal position to be within max speed bounds
			if not initialize:
				time_diff = timestamp - last_time
				last_time = timestamp
				# print("Time Diff:", time_diff)
				if np.linalg.norm(goal_position - cur_position) > max_speed*time_diff:
					print('Modifying goal_position from:', goal_position)
					goal_position = max_speed*time_diff*(goal_position - cur_position)/np.linalg.norm(goal_position - cur_position) + cur_position
					print('To:', goal_position)

			pose.translation = goal_position
			
			if initialize:                   
				# terminate active skills

				fa.goto_pose(pose)
				fa.goto_pose(pose, duration=T, dynamic=True, buffer_time=10, block = False,
					cartesian_impedances=[600.0, 600.0, 600.0, 10.0, 10.0, 10.0])
				initialize = False

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

			grip_wrapper.goto(goal_width, 0.15)
			i+=1
			rate.sleep()