import rospy
import cv2
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray
from sensor_msgs.msg import Image
from decoder import decodeImage
import time


LANE_DETECTION_NODE_NAME = 'lane_detection_node'
CAMERA_TOPIC_NAME = 'camera_rgb'
OBJECT_DETECTION_TOPIC = '/object_detection'
