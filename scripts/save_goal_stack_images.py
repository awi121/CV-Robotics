import cv2
import pickle
import argparse
import numpy as np
import pyrealsense2 as rs
import skimage.exposure
from block_segmentation_utils import *

W = 848
H = 480

# ----- Camera 3 (static) -----
REALSENSE_INTRINSICS_CAM_3 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_3 = "calib/realsense_camera2.tf"
parser = argparse.ArgumentParser()
parser.add_argument(
	"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS_CAM_3)
parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_TF_CAM_3)
args = parser.parse_args()
intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
transform = RigidTransform.load(args.extrinsics_file_path)
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322069488')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)
point_cloud = rs.pointcloud()

# align stream
aligned_stream = rs.align(rs.stream.color)

save_str = "simple_stack_target"
background_color = cv2.imread("scripts/Images/color_background.jpg")

for k in range(10):
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())
	depth_image = np.asanyarray(depth_frame.get_data())

	# skip empty frames
	if not np.any(depth_image):
		print("no depth")

	points = point_cloud.calculate(depth_frame)
	verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)

	# mask the color image to eliminate the background
	color_masked = remove_background(color_image, background_color) # masks out the background of the image

	# create contours
	clean_contours, clean_color_masked = generate_and_clean_contours(color_masked, color_image, H, W)

	# split into different block colors using LAB color space
	colors = ["green", "red", "blue", "beige"]
	blocks = identify_block_colors(color_image, clean_color_masked, verts, intrinsics, transform, colors, H, W, draw=False)
	stack_order = stack_order_heuristic(blocks)

	# save the data
	if k == 9:
		print("\nSAVE!!!\n")
		# save the block dict
		dict_file = "scripts/Images/" + save_str + "_dict.pkl"
		dict_pkl = open(dict_file, "wb")
		pickle.dump(blocks, dict_pkl)
		dict_pkl.close()

		# save the stack order
		stack_file = "scripts/Images/" + save_str + "_stack_order.pkl"
		stack_pkl = open(stack_file, "wb")
		pickle.dump(blocks, stack_pkl)
		stack_pkl.close()

		# save the original color image
		color_filename = "scripts/Images/" + save_str + ".jpg"
		cv2.imwrite(color_filename, color_image)

		# save the masked image
		color_filename = "scripts/Images/" + save_str + "_masked.jpg"
		cv2.imwrite(color_filename, clean_color_masked)

		# save the verts
		verts_reshape = verts.reshape(verts.shape[0], -1)
		verts_file = "scripts/Images/" + save_str + "_verts.txt"
		np.savetxt(verts_file, verts_reshape)

	# Show the images
	cv2.imshow("Color Image No Masking", color_image)
	cv2.imshow("Color Image Masked", clean_color_masked)
	cv2.waitKey(1)

