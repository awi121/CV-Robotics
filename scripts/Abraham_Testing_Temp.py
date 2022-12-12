from utils import *

a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

print(rotation_matrix_weighted_average([a, b], [0.5, 0.5]))




# from frankapy import FrankaArm, SensorDataMessageType
# from frankapy import FrankaConstants as FC
# from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
# from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
# from franka_interface_msgs.msg import SensorDataGroup
# import numpy as np
# import rospy

# fa = FrankaArm()
# fa.reset_joints()
# fa.close_gripper()
# pose = fa.get_pose()
# pose.translation = np.array([0.6, 0, 0.5])
# fa.goto_pose(pose)
# initialize = True

# i = 0
# dt = 0.02

# rate = rospy.Rate(1 / dt)
# pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
# T = 30

# z_path = np.linspace(pose.translation[2], 0.02, num=50)
# for z in z_path:
# 	pose.translation[2] = z
# 	print(z)
# 	if initialize:                   
# 		# terminate active skills

# 		# fa.goto_pose(pose)
# 		fa.goto_pose(pose, duration=T, dynamic=True, buffer_time=10,
# 			cartesian_impedances=[600.0, 600.0, 600.0, 10.0, 10.0, 10.0])
# 		initialize = False

# 		init_time = rospy.Time.now().to_time()
# 		timestamp = rospy.Time.now().to_time() - init_time
# 		last_time = timestamp
# 	else:
# 		timestamp = rospy.Time.now().to_time() - init_time
# 		traj_gen_proto_msg = PosePositionSensorMessage(
# 			id=i, timestamp=timestamp,
# 			position=pose.translation, quaternion=pose.quaternion
# 		)
# 		ros_msg = make_sensor_group_msg(
# 			trajectory_generator_sensor_msg=sensor_proto2ros_msg(
# 				traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
# 			)

# 		rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
# 		pub.publish(ros_msg)
# 		rate.sleep()

# fa.stop_skill()
# print('end')