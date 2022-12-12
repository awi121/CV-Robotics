import cv2
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

# align stream
aligned_stream = rs.align(rs.stream.color)
point_cloud = rs.pointcloud()

# ------ save baseline image -----
frames = pipeline.wait_for_frames()
frames = aligned_stream.process(frames)
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame().as_depth_frame()
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# skip empty frames
if not np.any(depth_image):
	print("no depth")

# edit the depth image to make it easier to visualize the depth
stretch = skimage.exposure.rescale_intensity(depth_image, in_range='image', out_range=(0,255)).astype(np.uint8)
stretch = cv2.merge([stretch, stretch, stretch])

background_color = color_image
background_depth = stretch

# # Show the images
# cv2.imshow("Color Image No Masking", color_image)
# cv2.imshow("Augmented Depth Image: ", stretch)
# cv2.waitKey(1)

color_filename = "scripts/Images/color_background.jpg"
depth_filename = "scripts/Images/depth_background.jpg"
cv2.imwrite(color_filename, color_image)
cv2.imwrite(depth_filename, depth_image)


# # import baseline background images
# background_color = cv2.imread("scripts/Images/color_background.jpg")

while True:
# for i in range(2):
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

	# define dictionary to keep track of blocks (key is the block number, value is [color, contour, (cX, cY), (X, Y, Z)])

	# split into different block colors using LAB color space
	colors = ["green", "red", "blue", "beige"]
	blocks = identify_block_colors(color_image, clean_color_masked, verts, intrinsics, transform, colors, H, W)

	# determine the number of individual blocks in the scene
	# print("\nTotal Blocks: ", len(blocks))
	stack_order = stack_order_heuristic(blocks)

	# determine the individual blocks' pixel COM
	for i in range(len(stack_order)):
		center = blocks[stack_order[i]][2]
		cX = center[0]
		cY = center[1]
		cv2.putText(color_image, str(i+1), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# Find matches for one block:
	if len(blocks) != 0:
		target_contour = blocks[stack_order[0]][0]
		target_color = blocks[stack_order[0]][1]

		matches = find_color_and_shape_match(target_contour, target_color, blocks)
		for match in matches:
			cX = match[2][0]
			cY = match[2][1]
			cv2.putText(color_image, "MATCH", (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# Show the images
	cv2.imshow("Color Image No Masking", color_image)
	cv2.imshow("Color Image Masked", clean_color_masked)
	cv2.waitKey(1)