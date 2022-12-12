import cv2
import numpy as np
import pyrealsense2 as rs
from block_segmentation_utils import *

# have two camera streams and perform feature matching between them!

W = 848
H = 480

REALSENSE_INTRINSICS_CAM_1 = "calib/realsense_intrinsics.intr"
REALSENSE_TF_CAM_1 = "calib/realsense_ee.tf"
pipeline_ee = rs.pipeline()
config_ee = rs.config()
config_ee.enable_device('220222066259')
config_ee.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_ee.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# ----- Camera 3 (static) -----
REALSENSE_INTRINSICS_CAM_3 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_3 = "calib/realsense_camera2.tf"
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322069488')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)
pipeline_ee.start(config_ee)

# align stream
aligned_stream = rs.align(rs.stream.color)
aligned_stream_ee = rs.align(rs.stream.color)

# # ------ save baseline image -----
# frames = pipeline.wait_for_frames()
# frames = aligned_stream.process(frames)
# color_frame = frames.get_color_frame()
# depth_frame = frames.get_depth_frame().as_depth_frame()
# color_image = np.asanyarray(color_frame.get_data())
# depth_image = np.asanyarray(depth_frame.get_data())

# # skip empty frames
# if not np.any(depth_image):
# 	print("no depth")

# # edit the depth image to make it easier to visualize the depth
# stretch = skimage.exposure.rescale_intensity(depth_image, in_range='image', out_range=(0,255)).astype(np.uint8)
# stretch = cv2.merge([stretch, stretch, stretch])

# background_color = color_image
# background_depth = stretch

# # # Show the images
# # cv2.imshow("Color Image No Masking", color_image)
# # cv2.imshow("Augmented Depth Image: ", stretch)
# # cv2.waitKey(1)

# color_filename = "scripts/Images/color_background.jpg"
# depth_filename = "scripts/Images/depth_background.jpg"
# cv2.imwrite(color_filename, color_image)
# cv2.imwrite(depth_filename, depth_image)


# import baseline background images
background_color = cv2.imread("scripts/Images/color_background.jpg")
background_depth = cv2.imread("scripts/Images/depth_background.jpg")



while True:
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())

	# get the ee frame to compare features between objects in the two frames!
	ee_frames = pipeline_ee.wait_for_frames()
	ee_frames = aligned_stream_ee.process(ee_frames)
	ee_color_frame = ee_frames.get_color_frame()
	ee_depth_frame = ee_frames.get_depth_frame().as_depth_frame()
	ee_color_image = np.asanyarray(ee_color_frame.get_data())
	# NOTE: maybe need to do the same background filtering w/ ee_image

	# eliminate the noisy background
	color_diff = abs(color_image - background_color)
	grey_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
	ret, foreground = cv2.threshold(grey_diff, 190, 255, cv2.THRESH_BINARY_INV)	# NOTE: maybe make this adaptive thresholding????
	# cv2.imshow("Foreground", foreground)

	# mask the color image
	color_masked = cv2.bitwise_and(color_image, color_image, mask=foreground)

	# create contours
	grey_masked = cv2.cvtColor(color_masked, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(grey_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cont_mask = np.zeros(grey_masked.shape, dtype='uint8')

	# generate clean contours (contours above certain area to eliminate noise)
	clean_contours = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			clean_contours.append(cnt)
	clean_contours = tuple(clean_contours)

	# convert clean contours into a new mask before color thresholding
	clean_mask = np.zeros((H,W), dtype='uint8')
	cv2.fillPoly(clean_mask, pts=clean_contours, color=(255,255,255))
	clean_color_masked = cv2.bitwise_and(color_image, color_image, mask=clean_mask)

	# define dictionary to keep track of blocks (key is the block number, value is [color, contour, (cX, cY), (X, Y, Z)])
	blocks = {}
	i = 0

	# convert to LAB color space
	lab_image = cv2.cvtColor(clean_color_masked, cv2.COLOR_BGR2LAB)
	l_channel = lab_image[:,:,0]
	a_channel = lab_image[:,:,1]	# spectrum from green to red
	b_channel = lab_image[:,:,2]	# spectrum from yellow to blue
	
	# green threshold (GOOD!)
	green_a_thresh = cv2.threshold(a_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	green_b_thresh = cv2.threshold(b_channel, 155, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	green_thresh = np.zeros((H,W), dtype='uint8')
	green_idx = np.where(np.equal(green_a_thresh, green_b_thresh))
	green_thresh[green_idx] = green_b_thresh[green_idx]
	green_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = green_thresh)
	green_cnts, green_hierarchy = cv2.findContours(cv2.cvtColor(green_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	blocks, i = iterate_contours(green_cnts, blocks, "green", color_image, i)

	# red threshold (GOOD!)
	red_a_thresh = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	red_b_thresh = cv2.threshold(b_channel, 100, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	red_thresh = np.zeros((H,W), dtype='uint8')
	red_idx = np.where(np.equal(red_a_thresh, red_b_thresh))
	red_thresh[red_idx] = red_a_thresh[red_idx]
	red_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = red_thresh)
	red_cnts, red_hierarchy = cv2.findContours(cv2.cvtColor(red_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	blocks, i = iterate_contours(red_cnts, blocks, "red", color_image, i)

	# blue threshold (GOOD ENOUGH! Eliminate beige with contour area constraints)
	blue_b_thresh = cv2.threshold(b_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	blue_l_thresh = cv2.threshold(l_channel, 90, 255, cv2.THRESH_BINARY_INV - cv2.THRESH_OTSU)[1]
	blue_thresh = np.zeros((H,W), dtype='uint8')
	blue_idx = np.where(np.equal(blue_b_thresh, blue_l_thresh))
	blue_thresh[blue_idx] = blue_b_thresh[blue_idx]
	blue_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = blue_thresh)
	blue_cnts, blue_hierarchy = cv2.findContours(cv2.cvtColor(blue_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	blocks, i = iterate_contours(blue_cnts, blocks, "blue", color_image, i)

	# beige threshold (GOOD!)
	beige_l_thresh = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	beige_white_thresh = cv2.threshold(l_channel, 125, 255, cv2.THRESH_BINARY_INV - cv2.THRESH_OTSU)[1] # this has been added to remove some white base still in image
	beige_a_thresh = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	beige_thresh_og = np.zeros((H,W), dtype='uint8')
	beige_idx = np.where(np.equal(beige_l_thresh, beige_a_thresh))
	beige_thresh_og[beige_idx] = beige_a_thresh[beige_idx]
	beige_thresh = np.zeros((H,W), dtype='uint8')
	beige_idx = np.where(np.equal(beige_thresh_og, beige_white_thresh))
	beige_thresh[beige_idx] = beige_thresh_og[beige_idx]
	beige_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = beige_thresh)
	beige_cnts, beige_hierarchy = cv2.findContours(cv2.cvtColor(beige_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	blocks, i = iterate_contours(beige_cnts, blocks, "beige", color_image, i)

	# determine the number of individual blocks in the scene
	print("\nTotal Blocks: ", len(blocks))
	stack_order = stack_order_heuristic(blocks)

	# determine the individual blocks' COM
	for i in range(len(stack_order)):
		center = blocks[stack_order[i]][2]
		cX = center[0]
		cY = center[1]
		cv2.putText(color_image, str(i+1), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# test matching the shapes of the contours
	block_0 = blocks[stack_order[0]]
	block_1 = blocks[stack_order[1]]
	block_2 = blocks[stack_order[2]]

	sim_00 = cv2.matchShapes(block_0[0], block_0[0], 1, 0.0)
	sim_01 = cv2.matchShapes(block_0[0], block_1[0], 1, 0.0)
	sim_12 = cv2.matchShapes(block_1[0], block_2[0], 1, 0.0)

	print("\nSim 00: ", sim_00)
	print("Sim 01: ", sim_01)
	print("Sim 12: ", sim_12)

	# TODOS: 
		# 1) calculate the COM along with drawing the contours and add that as an element of the dictionary value
		# 2) code up the heuristic based stacking function to plan the order to select the blocks
				# this function should return an ordered list of the block numbers (dictionary key) to pick up in that order
		# 3) classify each block's shape (use cv2.approxPolyDP to do this!) - this is naiive for 3D objects

	# # --------- test the comaprison with FLANN feature matching ----------
	# # step 1: detect the keypoints using SURF detector, compute the descriptors
	# max_features = 5500
	# good_match_pct = 0.03
	# # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=min_hessian) # NOTE: SURF algorithm not available w/ cv2
	# detector = cv2.SIFT_create(max_features)
	# img1 = cv2.cvtColor(clean_color_masked, cv2.COLOR_BGR2GRAY)
	# img2 = cv2.cvtColor(ee_color_image, cv2.COLOR_BGR2GRAY)
	# keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	# keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

	# # step 2: matching descriptor vctors with a FLANN based matcher
	# # since SURF is a floating-point descriptor, use NORM_L2
	# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
	# knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

	# # filter matches using the Lowe's ratio test
	# ratio_thresh = 0.7
	# good_matches = []
	# for m,n in knn_matches:
	# 	if m.distance < ratio_thresh * n.distance:
	# 		good_matches.append(m)

	# # draw the matches
	# img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
	# cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	# # show detected matches
	# cv2.imshow('Good Matches', img_matches)
	

	# Show the images
	cv2.imshow("Color Image No Masking", color_image)
	# cv2.imshow("EE Image: ", ee_color_image)
	# cv2.imshow("Color Image Masked", clean_color_masked)
	cv2.waitKey(1)