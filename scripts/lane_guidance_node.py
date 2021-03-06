#!/usr/bin/python3
import rospy
from std_msgs.msg import Float32, Int32, Int32MultiArray, Int16
import time

LANE_GUIDANCE_NODE_NAME = 'lane_guidance_node'
STEERING_TOPIC_NAME = '/steering'
THROTTLE_TOPIC_NAME = '/throttle'
CENTROID_TOPIC_NAME = '/centroid'
OBJECT_DETECTION_TOPIC = '/object_detection'


class PathPlanner:
    def __init__(self):
        # Initialize node and create publishers/subscribers
        self.init_node = rospy.init_node(LANE_GUIDANCE_NODE_NAME, anonymous=False)
        self.steering_publisher = rospy.Publisher(STEERING_TOPIC_NAME, Float32, queue_size=1)
        self.throttle_publisher = rospy.Publisher(THROTTLE_TOPIC_NAME, Float32, queue_size=1)
        self.object_detection_subscriber = rospy.Subscriber(OBJECT_DETECTION_TOPIC, Int16, self.object_detection_callback)
        self.steering_float = Float32()
        self.throttle_float = Float32()
        self.centroid_subscriber = rospy.Subscriber(CENTROID_TOPIC_NAME, Float32, self.controller)

        # Getting ROS parameters set from calibration Node
        self.steering_sensitivity = rospy.get_param('steering_sensitivity')
        self.no_error_throttle = rospy.get_param('no_error_throttle')
        self.error_throttle = rospy.get_param('error_throttle')
        self.error_threshold = rospy.get_param('error_threshold')
        self.zero_throttle = rospy.get_param('zero_throttle')
        self.stop = 0

        # # Display Parameters
        rospy.loginfo(
            f'\nsteering_sensitivity: {self.steering_sensitivity}'
            f'\nno_error_throttle: {self.no_error_throttle}'
            f'\nerror_throttle: {self.error_throttle}'
            f'\nerror_threshold: {self.error_threshold}'
            f'\nCatched_stop_value: {self.stop}')

    def controller(self, data):
        # try:
        kp = self.steering_sensitivity
        error_x = data.data
        rospy.loginfo(f"{error_x}")
        
        if self.stop == 1:
            throttle_float = self.zero_throttle
            time.sleep(0.003)
        elif error_x <= self.error_threshold:
            throttle_float = self.no_error_throttle
        else:
            throttle_float = self.error_throttle
        steering_float = float(kp * error_x)
        if steering_float < -1.0:
            steering_float = -1.0
        elif steering_float > 1.0:
            steering_float = 1.0
        else:
            pass
        self.steering_float.data = steering_float
        self.throttle_float.data = throttle_float
        self.steering_publisher.publish(self.steering_float)
        self.throttle_publisher.publish(self.throttle_float)
        rospy.on_shutdown(self.clean_shutdown)

    def clean_shutdown(self):
        print("Shutting Down...")
        self.throttle_float.data = self.zero_throttle
        self.throttle_publisher.publish(self.throttle_float)
        print("Shut down complete")

    def object_detection_callback(self, data):
        self.stop = int(data.data)
        # print("Catched Value: "+str(self.stop))


def main():
    path_planner = PathPlanner()
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()


if __name__ == '__main__':
    main()
