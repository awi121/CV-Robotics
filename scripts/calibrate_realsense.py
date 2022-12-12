import cv2
import numpy as np
import pyrealsense2 as rs
# from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

"""
This script calibrates a realsense camera in the robot's world frame. Specifically,
this script outputs the camera extrinsics (translation and rotation from camera to
robot) and the camera intrinsics.
"""

# ------ Calirbation Information -------
# Camera 2: '151322066099', checkerboard = (5,12)
# Camera 3: '151322069488', checkerboard = (5,16)
# Camera 4: '151322061880', checkerboard = (5,26)
# Camera 5: '151322066932', checkerboard = (5,9)

# name of camera to label extrinsic and intrinsic save files 
save_name = "realsense_static_1"

# define the size of checkerboard being used to calibrate 
# NOTE: need to check to make sure board of desired dimension is in frame
checkerboard = (5,9) # (5,26) # (5,8) # extended is (5,17)

# lists of points
objpoints = [] # 3d real-world point in robot frame
imgpoints = [] # 2d pixel point in image plane

# ground truth x,y,z position
pose3d = []

# NOTE: may be generating this wrong!!!!!!

# NOTE: this is assuming going from farthest to closest, right to left
# for i in reversed(range(1,checkerboard[1]+1)):
for i in reversed(range(27-checkerboard[1], 27)):
# for i in range(1,checkerboard[1]+1):
	# print(i)
	x = 0.1 + i*0.028
	# x = 0.34 + i*0.028

	for j in reversed(range(-2,3)):
	# for j in range(-2,3):
		# print("J: ", j)
		y = j*0.028
		pose3d.append([x,y,0])

pose3d = np.array(pose3d, np.float32)

print("\nPose 3d: ", pose3d)

# criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Configure depth and color streams
W = 848
H = 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322066932') # decide which camera to calibrate
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

pipeline.start(config)
aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

for i in range(500): # NOTE: 500 iterations is best!
	# get an image from the camera
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())
	bw_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	# find checkerboard corners
	ret, corners = cv2.findChessboardCorners(bw_image, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
	# print("\nCorners: ", corners)
	if ret == True:

		corners_refined = cv2.cornerSubPix(bw_image, corners, (11, 11), (-1, -1), criteria)

		# # visualize the corners in the order they appear (for debugging purposes)
		# for corner in corners:
		# 	# print("Corner: ", corner)
		# 	cX = int(corner[0][0])
		# 	cY = int(corner[0][1])
		# 	cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
		# 	cv2.imshow('image', color_image)
		# 	cv2.waitKey(220)

		imgpoints.append(np.array(corners_refined.reshape(corners_refined.shape[0],2), np.float32))
		objpoints.append(pose3d)

	print("Iteration: ", i)

# NOTE: maybe add a second loop that waits for a keypress to allow one to move the checkerboard (i.e. lift it to vary z)


ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, bw_image.shape[::-1], None, None)

print("\nCamera Matrix: ", matrix)

print("\nDistortion: ", distortion)

# convert from rotation vector to rotation matrix
rotation_mat, _ = cv2.Rodrigues(np.mean(r_vecs, axis=0))

# get mean of translation vector
tvec_mean = np.mean(t_vecs, axis=0)

print("\nOriginal Translation: ", tvec_mean)
print("\nOriginal Rotation Matrix: ", rotation_mat)

translation = np.array([tvec_mean[2], tvec_mean[0], -tvec_mean[1]])
rotation = np.linalg.inv(rotation_mat)
transformation = np.concatenate((np.transpose(translation), rotation), axis=0)

print("\nTranslation Vector: ", translation)
print("\nRotation Matrix: ", rotation)
print("\nTransformation Matrix: ", transformation)

# # save data to text file for now!
# filename = save_name + ".txt"
# file = open(filename, "x")
# file.write("realsense_ee")
# file.write("\nfranka_tool")
# file.close()


# print("\nRotation Matrix Average: ", rotation_mat)

# print("\nTranslation Average: ", np.mean(t_vecs, axis=0))

# # swap axis the align with world frame correctly
# rvec_mean = np.mean(r_vecs, axis=0)
# print("\nAverage rotation vector: ", rvec_mean)
# print("\nUpdated Frame Rotation Vector: ", np.array([rvec_mean[2], rvec_mean[0], -rvec_mean[1]]))

# new_mat, _ = cv2.Rodrigues(np.array([rvec_mean[2], rvec_mean[0], -rvec_mean[1]]))
# print("\nUpdated Rotation Matrix: ", new_mat)


# print("\nUpdated Frame Translation: ", np.array([tvec_mean[2], tvec_mean[0], -tvec_mean[1]]))

# print("\nInverse Original Rotation: ", np.linalg.inv(rotation_mat))



# print("\nInverse Original Translation: ", np.linalg.inv(tvec_mean))


# TODO: automatically generate the .tf and .intr files with given names when running the calibration script