# import necessary packages and files
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

# REALSENSE_INTRINSICS = "calib/realsense_static_intrinsics.intr"
# REALSENSE_EE_TF = "calib/realsense_static.tf"

class DetectObject:
	"""
	This is an object detection class that given an object, it will 
	predict the object's 6D pose. The object can be detected using
	AprilTags, or CNN methods. The pose prediction can be computed by 
	either the wrist-mounted camera, the statically-mounted camera, 
	or both. 
	"""
	def __init__(self, object_id, object_class):
	# def __init__(self, realsense_intrinsics_ee, realsense_to_ee_transform, realsense_intrinsics_static, realsense_to_static_transform, object_id, object_class):
		"""
		Initialize DetectObject.

		NOTE: currently not doing anything with object_id or object_class, instead will soon be 
		parameters size, color, type, apriltags

		Parameters
		----------
		object_id: ID number of the detected object
		object_class: Class of the detected object 
		"""
		# load in arguments
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
		)
		parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
		args = parser.parse_args()

		self.realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
		self.realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)


		# self.realsense_intrinsics_ee = realsense_intrinsics_ee
		# self.realsense_to_ee_transform = realsense_to_ee_transform
		# self.realsense_intrinsics_static = realsense_intrinsics_static
		# self.realsense_to_static_transform = realsense_to_static_transform
		self.object_id = object_id
		self.object_class = object_class 

		# initialize object position, velocity, rotation, angular velocity all to zero
		self.object_center_point = np.array([0,0,0])
		self.object_velocity = np.array([0,0,0])
		self.object_rotation = np.array([0,0,0,0])
		self.object_ang_velocity = np.array([0,0,0])

		# size, color, type, block offset (NOTE: these should be inputs to the init file)
		self.size = "0.03,0.03,0.03" #[m]
		self.color = "0,0,255,1"
		self.type = "block"
		self.center_offset_vector = np.array([[0],[0],[0.015]])

	def _return_current_position(self):
		"""
		Returns the current object x,y,z position.

		Returns
		-------
		the object center point
		"""
		return self.object_center_point

	def _return_current_velocity(self):
		"""
		Returns the current object translational velocity.

		Returns
		-------
		the object translational velocity
		"""
		return self.object_velocity

	def _return_current_rotation(self):
		"""
		Returns the current object rotation.

		Returns
		-------
		the object rotation
		"""
		return self.object_rotation

	def _return_current_ang_velocity(self):
		"""
		Returns the current object angular velocity.

		Returns
		-------
		the object angular velocity
		"""
		return self.object_ang_velocity
	
	def _return_size(self):
		"""
		Returns the size of the object.

		Returns
		-------
		the object center point
		"""
		return self.size 

	def _return_color(self):
		"""
		Returns the color of the object.

		Returns
		-------
		the object color
		"""
		return self.color 

	def _return_type(self):
		"""
		Returns the type of the object.

		Returns
		-------
		the object type
		"""
		return self.type

	def _get_positions_depth_nodepth(self, bounds, verts, intrinsics, transform, robot_pose):
		"""
		INSERT EXPLANATION
		"""

		# ---- Depth-Based Prediction ----
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

		median_point = np.array([x_pos, y_pos, z_pos])
		object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, intrinsics, transform, robot_pose)
		com_depth = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])

		# ---- Image-Based Prediction (No Depth) ----
		com_nodepth = get_object_center_point_in_world_realsense_3D_camera_point(translation_matrix, intrinsics, transform, robot_pose)
		com_nodepth = np.array([com_nodepth[0], com_nodepth[1], com_nodepth[2]])

		return com_depth, com_nodepth

	# def get_position_apriltag(self, bounds, verts, robot_pose, translation_matrix, static, gipper, rotation_matrix=None):
	def get_position_apriltag(self, bounds, verts, robot_pose, translation_matrix, rotation_matrix=None):
		"""
		Estimate the object position in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
		----------
		bounds: numpy array of the bounding box of the detected object
		verts: pointcloud of the scene 
		robot_pose: the pose of the robot end-effector
		translation_matrix: translation of apriltag in camera frame
		rotation_matrix: rotation of apriltag in camera frame
			if None, find the center of the tag, not the object
		normal_offset_vector: vector displacemetn of the center point 
			from the center of the tag, in the tag's coordinate frame
			(ie. for the block, the vector is [0, 0, -0.015] since the center 
			is 1.5cm into the block (normal to the tag) and centered on the tag.)
			if None, find the center of the tag, not the object

		Returns
		-------
		object_center_point: the x,y,z coordinate of the center of the object
			in the robot's coordinate frame
		"""


		# # ---- Determine How Many Cameras ----
		# # only the static camera detected this object
		# if static and not gripper:
		# 	bounds = bounds[0]
		# 	verts = verts[0]
		# 	com_depth, com_nodepth = self._get_positions_depth_nodepth(bounds, verts, self.realsense_intrinsics_static, self.realsense_to_static_transform, robot_pose)

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth.any()):
		# 		com_depth = com_nodepth
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
		# 		com_depth[2] = com_nodepth[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()

		# # only gripper camera detected this object
		# elif gripper and not static:
		# 	bounds = bounds[0]
		# 	verts = verts[0]
		# 	com_depth, com_nodepth = self._get_positions_depth_nodepth(bounds, verts, self.realsense_intrinsics_ee, self.realsense_to_ee_transform, robot_pose)
		# 	com_nodepth[2]+=0.03

		# 	# scale the no-depth y estimate to account for some linear error we determined experimentally
		# 	delta_y = -0.22*com_depth[2] + 0.11
		# 	com_nodepth[1]-=delta_y

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth.any()):
		# 		com_depth = com_nodepth
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
		# 		com_depth[2] = com_nodepth[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()

		# # both cameras detected this object
		# else:
		# 	com_depth_static, com_nodepth_static = self._get_positions_depth_nodepth(bounds[0], verts[0], self.realsense_intrinsics_static, self.realsense_to_static_transform, robot_pose)
		# 	com_depth_ee, com_nodepth_ee = self._get_positions_depth_nodepth(bounds[1], verts[1], self.realsense_intrinsics_ee, self.realsense_to_ee_transform, robot_pose)
		# 	com_nodepth_ee[2]+=0.03

		# 	# scale the no-depth y estimate to account for some linear error we determined experimentally
		# 	delta_y = -0.22*com_nodepth_ee[2] + 0.11
		# 	com_nodepth_ee[1]-=delta_y

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth_static.any()):
		# 		com_depth_static = com_nodepth_static
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth_static[2] - com_nodepth_static[2]) > 0.1:
		# 		com_depth_static[2] = com_nodepth_static[2]
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth_ee.any()):
		# 		com_depth_ee = com_nodepth_ee
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth_ee[2] - com_nodepth_ee[2]) > 0.1:
		# 		com_depth_ee[2] = com_nodepth_ee[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth_static[0] + com_nodepth_static[0] + com_depth_ee[0] + com_nodepth_ee[0])/4, (com_depth_static[1] + com_nodepth_static[1] + com_depth_ee[1] + com_nodepth_ee[1])/4, (2*com_depth_static[2] + com_nodepth_static[2] + 2*com_depth_ee[2] + com_nodepth_ee[2])/6])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()











		# ORIGINAL CODE:

		# ---- Depth-Based Prediction ----
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

		print("\nDetect Object Extrinsics: ", self.realsense_to_ee_transform)

		median_point = np.array([x_pos, y_pos, z_pos])

		object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)
		com_depth = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])

		print("\nCOM Depth: ", com_depth)

		# ---- Image-Based Prediction (No Depth) ----
		com_nodepth = get_object_center_point_in_world_realsense_3D_camera_point(translation_matrix, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)
		com_nodepth = np.array([com_nodepth[0], com_nodepth[1], com_nodepth[2]])
		# com_nodepth[2]+=0.03

		# # scale the no-depth y estimate to account for some linear error we determined experimentally
		# delta_y = -0.22*com_depth[2] + 0.11
		# com_nodepth[1]-=delta_y

		print("No Depth: ", com_nodepth)

		# ---- Combine Predictions ----
		# if depth-based prediction is Nan, only use non-depth-based prediction
		if np.isnan(com_depth.any()):
			com_depth = com_nodepth
		# if the prediction difference between depth and no depth is large ignore depth-based z
		elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
			com_depth[2] = com_nodepth[2]

		# weighted average
		self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
		
		# BEGIN 10/26 ROTATION EDIT
		if (rotation_matrix is not None and self.center_offset_vector is not None):
			disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
			self.object_center_point = self.object_center_point + disp_vector.flatten()

		# convert from camera frame to world frame
		# END 10/26 ROTATION EDIT

		return self.object_center_point

	def get_position_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object position in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
		----------
		color_image: opencv image of the workspace
		depth_image: opencv depth image of the workspace
		object_bounds: numpy array of the bounding box of the detected object
		robot_pose: the pose of the robot end-effector

		Returns
		-------
		object_center_point: the x,y,z coordinate of the center of the object
			in the robot's coordinate frame
		"""

		# crop the image and depth information using the object_bounds -- actually mask image to preserve original image size!!!!

		# TODO: make the z position robust to depth uncertainty!!!!!
			# store up the depth prediction from multiple points/multiple frames & get the average
			# get the depth prediction from mutliple points on surface and if the variance is too high, scrap it
			# get the depth info and if the difference from previous prediction is too big, then ignore?

		blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# adaptive thresholding on greyscale image
		gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		kernal = np.ones((5,5), "uint8")
		gray_mask = cv2.dilate(thresh, kernal)

		# create contours
		contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cont_mask = np.zeros(gray_mask.shape, dtype='uint8')

		# draw/calculate the centers of objects
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 800:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				width = int(np.sqrt(area)/8)


				# TODO: Generating samples not robust with no guarantees this isn't larger than the image!!!!
				width = 5
				pixel_pairs = []
				cx_start = cX - width
				cy_start = cY - width
				for i in range(2*width):
					x = cx_start + i
					for j in range(2*width):
						y = cy_start + j
						pixel_pairs.append([x,y])

				object_center_point_in_world, variance = get_object_center_point_in_world_realsense_robust(
					cX,
					cY,
					pixel_pairs,
					depth_image,
					self.realsense_intrinsics,
					self.realsense_to_ee_transform,
					robot_pose)

				# if variance is too high, then ignore z position update
				if variance > 1e-4:
					print("high variance....ignore z update")
					self.object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], self.object_center_point[2]])
				else:
					self.object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])

		return self.object_center_point

	def get_color_image(self, color_image, depth_image, object_bounds):
		"""
		Get the color of the object from image input.

		Parameters
		----------
		color_image: opencv image of the workspace
		depth_image: opencv depth image of the workspace
		object_bounds: numpy array of the bounding box of the detected object

		Returns
		-------
		color: the color of the object (assuming consistent color)
		"""
		pass

	def get_orientation_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object pose in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
		----------
		color_image: opencv image of the workspace
		depth_image: opencv depth image of the workspace
		object_bounds: numpy array of the bounding box of the detected object
		robot_pose: the pose of the robot end-effector

		Returns
		-------
		object_center_pose: the pose of the center of the object
			in the robot's coordinate frame
		"""
		pass 

	def get_pose_apriltag(self, bounds, verts, robot_pose, translation_matrix, rotation_matrix=None):
		"""
		Estimate the object pose in the robot's frame given the image, depth,
		object bounds, current robot pose based on AprilTag detection, and 
		object center offset vecotor, which point.

		Parameters
		----------
		bounds: numpy array of the bounding box of the detected object
		verts: pointcloud of the scene 
		robot_pose: the pose of the robot end-effector
		translation_matrix: translation of apriltag in camera frame
		rotation_matrix: rotation of apriltag in camera frame
			if None, find the center of the tag, not the object

		Returns
		-------
		object_center_point: the x,y,z coordinate of the center of the object
			in the robot's coordinate frame
		"""


		# # ---- Determine How Many Cameras ----
		# # only the static camera detected this object
		# if static and not gripper:
		# 	bounds = bounds[0]
		# 	verts = verts[0]
		# 	com_depth, com_nodepth = self._get_positions_depth_nodepth(bounds, verts, self.realsense_intrinsics_static, self.realsense_to_static_transform, robot_pose)

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth.any()):
		# 		com_depth = com_nodepth
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
		# 		com_depth[2] = com_nodepth[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()

		# # only gripper camera detected this object
		# elif gripper and not static:
		# 	bounds = bounds[0]
		# 	verts = verts[0]
		# 	com_depth, com_nodepth = self._get_positions_depth_nodepth(bounds, verts, self.realsense_intrinsics_ee, self.realsense_to_ee_transform, robot_pose)
		# 	com_nodepth[2]+=0.03

		# 	# scale the no-depth y estimate to account for some linear error we determined experimentally
		# 	delta_y = -0.22*com_depth[2] + 0.11
		# 	com_nodepth[1]-=delta_y

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth.any()):
		# 		com_depth = com_nodepth
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
		# 		com_depth[2] = com_nodepth[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()

		# # both cameras detected this object
		# else:
		# 	com_depth_static, com_nodepth_static = self._get_positions_depth_nodepth(bounds[0], verts[0], self.realsense_intrinsics_static, self.realsense_to_static_transform, robot_pose)
		# 	com_depth_ee, com_nodepth_ee = self._get_positions_depth_nodepth(bounds[1], verts[1], self.realsense_intrinsics_ee, self.realsense_to_ee_transform, robot_pose)
		# 	com_nodepth_ee[2]+=0.03

		# 	# scale the no-depth y estimate to account for some linear error we determined experimentally
		# 	delta_y = -0.22*com_nodepth_ee[2] + 0.11
		# 	com_nodepth_ee[1]-=delta_y

		# 	# ---- Combine Predictions ----
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth_static.any()):
		# 		com_depth_static = com_nodepth_static
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth_static[2] - com_nodepth_static[2]) > 0.1:
		# 		com_depth_static[2] = com_nodepth_static[2]
		# 	# if depth-based prediction is Nan, only use non-depth-based prediction
		# 	if np.isnan(com_depth_ee.any()):
		# 		com_depth_ee = com_nodepth_ee
		# 	# if the prediction difference between depth and no depth is large ignore depth-based z
		# 	elif abs(com_depth_ee[2] - com_nodepth_ee[2]) > 0.1:
		# 		com_depth_ee[2] = com_nodepth_ee[2]

		# 	# weighted average
		# 	self.object_center_point = np.array([(com_depth_static[0] + com_nodepth_static[0] + com_depth_ee[0] + com_nodepth_ee[0])/4, (com_depth_static[1] + com_nodepth_static[1] + com_depth_ee[1] + com_nodepth_ee[1])/4, (2*com_depth_static[2] + com_nodepth_static[2] + 2*com_depth_ee[2] + com_nodepth_ee[2])/6])
			
		# 	# BEGIN 10/26 ROTATION EDIT
		# 	if (rotation_matrix is not None and self.center_offset_vector is not None):
		# 		disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
		# 		self.object_center_point = self.object_center_point + disp_vector.flatten()











		# ORIGINAL CODE:

		# ---- Depth-Based Prediction ----
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

		print("\nDetect Object Extrinsics: ", self.realsense_to_ee_transform)

		median_point = np.array([x_pos, y_pos, z_pos])

		object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)
		com_depth = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])

		print("\nCOM Depth: ", com_depth)

		# ---- Image-Based Prediction (No Depth) ----
		com_nodepth = get_object_center_point_in_world_realsense_3D_camera_point(translation_matrix, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)
		com_nodepth = np.array([com_nodepth[0], com_nodepth[1], com_nodepth[2]])
		# com_nodepth[2]+=0.03

		# # scale the no-depth y estimate to account for some linear error we determined experimentally
		# delta_y = -0.22*com_depth[2] + 0.11
		# com_nodepth[1]-=delta_y

		print("No Depth: ", com_nodepth)

		# ---- Combine Predictions ----
		# if depth-based prediction is Nan, only use non-depth-based prediction
		if np.isnan(com_depth.any()):
			com_depth = com_nodepth
		# if the prediction difference between depth and no depth is large ignore depth-based z
		elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
			com_depth[2] = com_nodepth[2]

		# weighted average
		self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])
		
		# BEGIN 10/26 ROTATION EDIT
		if (rotation_matrix is not None and self.center_offset_vector is not None):
			disp_vector = robot_pose.rotation@self.realsense_to_ee_transform.rotation@rotation_matrix@self.center_offset_vector
			self.object_center_point = self.object_center_point + disp_vector.flatten()

		# convert from camera frame to world frame
		# END 10/26 ROTATION EDIT

		return self.object_center_point
	def get_pose_apriltag(self, tag_rotation_matrix, initial_rotation):
		"""
		Estimate the object pose in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
		----------
		tag_rotation_matrix: the rotation matrix of the apriltag
		initial_rotation: the initial rotation of the tag corresponding to 0 in all axes

		Returns
		-------
		object_center_pose: the pose of the center of the object
			in the robot's coordinate frame
		"""
		pass 

	def get_velocity_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object velocity in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
		----------
		color_image: opencv image of the workspace
		depth_image: opencv depth image of the workspace
		object_bounds: numpy array of the bounding box of the detected object
		robot_pose: the pose of the robot end-effector

		Returns
		-------
		object_velocity: the velocity of the center of the object
			in the robot's coordinate frame
		"""
		pass 

	def get_velocity_apriltag(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object velocity in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
		----------
		color_image: opencv image of the workspace
		depth_image: opencv depth image of the workspace
		object_bounds: numpy array of the bounding box of the detected object
		robot_pose: the pose of the robot end-effector

		Returns
		-------
		object_velocity: the velocity of the center of the object
			in the robot's coordinate frame
		"""
		pass