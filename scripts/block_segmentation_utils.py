import cv2
import numpy as np
import pyrealsense2 as rs
from utils import *

def remove_background(image, background_image):
	"""
	Given a background image (image without any objects in the environment), and the
	current image, mask out the background.

	:param image: 				the RGB image of the scene
	:param background_image: 	the RGB iamge of the scene without any objects
	:return: 					RGB image with background masked out
	"""
	color_diff = abs(image - background_image)
	grey_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
	ret, foreground = cv2.threshold(grey_diff, 190, 255, cv2.THRESH_BINARY_INV)
	image_masked = cv2.bitwise_and(image, image, mask=foreground)
	return image_masked

def generate_and_clean_contours(image_masked, image, H, W):
	"""
	Given an image with the background masked out, find the contours of all the objects
	in the scene, and clean it with thresholding by contour area. Additionally, generate
	new masked image leaving only the clean contours visible.

	:param image_masked: 	RGB image with background masked out
	:param image: 			original RBG image of the scene
	:return: 				list of clean contours, cleaned up masked image showing the shapes
	"""
	grey_masked = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)
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
			# cv2.drawContours(image, [cnt], 0, (0,0,255), 2)
			clean_contours.append(cnt)

	clean_contours = tuple(clean_contours)

	# convert clean contours into a new mask before color thresholding
	clean_mask = np.zeros((H,W), dtype='uint8')
	cv2.fillPoly(clean_mask, pts=clean_contours, color=(255,255,255))
	clean_color_masked = cv2.bitwise_and(image, image, mask=clean_mask)
	return clean_contours, clean_color_masked

def mask_color(image_masked, H, W, thresh1, thresh2, thresh3, flag):
	"""
	INSERT
	"""
	if flag:
		thresh_intermediate = np.zeros((H,W), dtype='uint8')
		idx = np.where(np.equal(thresh1, thresh2))
		thresh_intermediate[idx] = thresh1[idx]
		thresh = np.zeros((H,W), dtype='uint8')
		idx = np.where(np.equal(thresh_intermediate, thresh3))
		thresh[idx] = thresh_intermediate[idx]
	else:
		thresh = np.zeros((H,W), dtype='uint8')
		idx = np.where(np.equal(thresh1, thresh2))
		thresh[idx] = thresh1[idx]
	return thresh

def get_lab_color_thresholds(color, l_channel, a_channel, b_channel):
	"""
	INSERT
	"""
	if color == "green":
		thresh1 = cv2.threshold(a_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		thresh2 = cv2.threshold(b_channel, 155, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		return thresh1, thresh2, None, False
	elif color == "red":
		thresh1 = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		thresh2 = cv2.threshold(b_channel, 100, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		return thresh1, thresh2, None, False
	elif color == "blue":
		thresh1 = cv2.threshold(b_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		thresh2 = cv2.threshold(l_channel, 90, 255, cv2.THRESH_BINARY_INV - cv2.THRESH_OTSU)[1]
		return thresh1, thresh2, None, False
	elif color == "beige":
		thresh1 = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		thresh2 = cv2.threshold(l_channel, 125, 255, cv2.THRESH_BINARY_INV - cv2.THRESH_OTSU)[1] # this has been added to remove some white base still in image
		thresh3 = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
		return thresh1, thresh2, thresh3, True
	else:
		print("\nERROR: no pre-defined color thresholds for that color!")
		return None, None, None

def identify_block_colors(image, image_masked, verts, intrinsics, transform, colors, H, W, draw=True):
	"""
	INSERT
	"""
	blocks = {}
	i = 0

	# convert to LAB color space
	lab_image = cv2.cvtColor(image_masked, cv2.COLOR_BGR2LAB)
	l_channel = lab_image[:,:,0]
	a_channel = lab_image[:,:,1]	# spectrum from green to red
	b_channel = lab_image[:,:,2]	# spectrum from yellow to blue

	for color in colors:
		thresh1, thresh2, thresh3, flag = get_lab_color_thresholds(color, l_channel, a_channel, b_channel)
		thresh = mask_color(image_masked, H, W, thresh1, thresh2, thresh3, flag)
		masked = cv2.bitwise_and(image_masked, image_masked, mask = thresh)
		cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.imshow(color, masked)
		blocks, i = iterate_contours(cnts, blocks, color, image, verts, intrinsics, transform, i, draw)
	return blocks

def get_contour_position(contour, verts, intrinsics, transform):
	"""
	INSERT
	"""
	# ---- Depth-Based Prediction -----
	contour_array = np.squeeze(np.array(contour))
	minx = np.amin(contour_array[:,0])
	maxx = np.amax(contour_array[:,0])
	miny = np.amin(contour_array[:,1])
	maxy = np.amax(contour_array[:,1])
	
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
	object_median_point = get_object_center_point_in_world_realsense_static_camera(median_point, intrinsics, transform)
	com_depth = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])
	return com_depth


def iterate_contours(contours, block_dict, color, color_image, verts, intrinsics, transform, i, draw=True):
	"""
	Iterate through the detected contours, determine the center pixel of the contour,
	draw the cnotour on the image, and add the contour properties to the block dictionary.

	:param contours: 	the contours associated with a specific color
	:param block_dict: 	the dictionary that assigns each new block a unique ID as the key
	:param color: 		string of the color of the block
	:param color_image: the image to display the contour on
	:param i: 			the contour number overall
	:returns: 			updated block_dict and i
	"""
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			# cnt_approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
			# cv2.drawContours(color_image, [cnt_approx], 0, (0,0,255), 2)
			if draw:
				cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			# cv2.putText(color_image, color, (cX - 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			# determine block COM in robot frame
			com = get_contour_position(cnt, verts, intrinsics, transform)

			# adjustments to com
			com[0]+=0.37
			com[1]+=0.2
			com[2]+=1.3

			if draw:
				center_text = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(com[0], com[1], com[2])
				# center_text = "(" + str(com[0]) + ", " + str(com[1]) + ", " + str(com[2]) + ")"
				cv2.putText(color_image, center_text, (cX + 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			block_dict[i] = [cnt, color, (cX, cY), com]
			i+=1

	return block_dict, i

def match_shape(target_cnt, blocks):
	"""
	Given a target contour, compare to all contours in contours and return those that return an
	accuracy above designated threshold.

	:param target_cnt: 	the contour that we are trying to match the shape to
	:param blocks:		the dictionary of all blocks in the scene
	:return:			matches
	"""
	matches = []
	for block in blocks:
		cnt = blocks[block][0]
		diff = cv2.matchShapes(cnt, target_cnt, 1, 0.0)
		if diff < 0.1:
			matches.append(blocks[block])
	return matches 

def match_color(target_color, block_list):
	"""
	Given a target color, compare all colors.

	:param target_color: 	the color that we are trying to match
	:param block_list:		list of all blocks in the scene
	:return:				matches
	"""
	matches = []
	for block in block_list:
		color = block[1]
		if color == target_color:
			matches.append(block)
	return matches

def find_color_and_shape_match(target_cnt, target_color, blocks):
	"""
	Given a target contour shape and color, find any contours matching both properties.

	:param target_ctn: 		the contour that we are trying to match the shape to
	:param target_color:	the color that we are trying to match
	:param blocks: 			the dictionary of all blocks in the scene
	:return:				matches
	"""
	shape_matches = match_shape(target_cnt, blocks)
	color_shape_matches = match_color(target_color, shape_matches)

	if len(color_shape_matches) == 0:
		print("\nWARNING: Did not find any shape and color matches!")

	return color_shape_matches

def stack_order_heuristic(blocks):
	"""
	This function returns the stacking order of the blocks based on their pixel locations.
	Stack order is determined by lowest y values, and then left to right.

	:param blocks: a dictionary with the keys being each block's identifier and the value being the
				   block's color, and contours
	:returns: list of the block keys in the order for them to be placed
	"""

	# NOTE: smaller values of pixel Y is higher up in image, smaller values of pixel X is left in image
	# prioritize largest Y's first - if two Y's are similar, prioritize smaller X
	# IDEA: create np array and argsort based on the Y??

	y_vals = np.zeros(len(blocks))
	for block in blocks:
		y_vals[block] = blocks[block][2][1]

	# naiive approach (only care about y order and random x)
	return np.argsort(-y_vals)
