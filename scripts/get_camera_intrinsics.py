import pyrealsense2 as rs

# pipe = rs.pipeline
# pipe_profile = pipe.start()
# intrinsics = pipe_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
# print(intrinsics)


# config.enable_device('151322061880')

# cfg = pipeline.start()
W = 848
H = 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322061880')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.depth)
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)