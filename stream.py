import cv2
import pyrealsense2 as rs
import open3d
import numpy as np

W = 640
H = 480
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.infrared)
# pipeline.start(config)
# frames = pipeline.wait_for_frames()
# ir = frames.first(rs.stream.infrared)
config.enable_device('151322066099')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

# get a stream of images
while True:
# for i in range(50):
    # ----- added from other method
    # current_pose = fa.get_pose()
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame().as_depth_frame()

    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
   
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())

    # skip empty frames
    if not np.any(depth_image):
        print("no depth")
        # continue

    # print("\n[INFO] found a valid depth frame")
    color_image = np.asanyarray(color_frame.get_data())
    bw_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # camera parameters [fx, fy, cx, cy]
    # cam_param = [realsense_intrinsics.fx, realsense_intrinsics.fy, realsense_intrinsics.cx, realsense_intrinsics.cy]


    cv2.imshow("Image", color_image)
    k=cv2.waitKey(1)
    if(k==27):
        # Wait for the next set of frames from the camera
        # colorized = colorizer.process(frames)
        points.export_to_ply("2.ply", color_frame)

# Create save_to_ply object
        # ply = rs.save_to_ply("1.ply")

# Set options to the desired values
# In this example we'll generate a textual PLY with normals (mesh is already created by default)
        # ply.set_option(rs.save_to_ply.option_ply_binary, False)
        # ply.set_option(rs.save_to_ply.option_ply_normals, True)

        print("Saving to 1.ply...")
# Apply the processing block to the frameset which contains the depth frame and the texture
        # ply.process(colorized)
        print("Done")

        break
          
