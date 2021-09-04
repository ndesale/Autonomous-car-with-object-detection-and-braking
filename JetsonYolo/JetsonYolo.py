import cv2 
import numpy as np 
from elements.yolo import OBJ_DETECTION 

import rospy
from std_msgs.msg import Int32, Int32MultiArray, String, Int16
from sensor_msgs.msg import Image
#from decoder import decodeImage
import time
from cv_bridge import CvBridge


OBJECT_DETECTION_NODE = 'object_detection_node'
CAMERA_TOPIC_NAME = 'camera_rgb'
OBJECT_DETECTION_TOPIC = '/object_detection'
CAMERA_TOPIC_NAME = 'camera_rgb'


Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',                'hair drier', 'toothbrush' ] 

custom_list = ['person', 'dog', 'cat', 'horse', 'pizza']

Object_colors = list(np.random.rand(80,3)*255) 
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes) 

window_handle = cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)

class ObjectDetection:

    def __init__(self):
        self.init_node = rospy.init_node(OBJECT_DETECTION_NODE, anonymous=False)
        self.object_detection_publisher = rospy.Publisher(OBJECT_DETECTION_TOPIC, Int16, queue_size=1)
        self.camera_subscriber = rospy.Subscriber(CAMERA_TOPIC_NAME, Image, self.DetectObjects)
        self.bridge = CvBridge()

    def DetectObjects(self, data):
        frame = self.bridge.imgmsg_to_cv2(data)
        
        if True:
            #window_handle = cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE) 
            # Window
            if True:    # cv2.getWindowProperty("USB Camera", 0) >= 0: 
                ret = True
                if ret and frame is not None:
                        # detection process 
                        objs = Object_detector.detect(frame) 

                        # plotting 
                        for obj in objs: 
                                # print(obj) 
                                label = obj['label'] 
                                score = obj['score'] 
                                [(xmin,ymin),(xmax,ymax)] = obj['bbox'] 
                                color = Object_colors[Object_classes.index(label)] 
                                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                                frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin),
                                        cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA) 

                                if label in custom_list:
                                    self.object_detection_publisher.publish(1)
                                    print(label)
                                else:
                                    self.object_detection_publisher.publish(0)

                #cv2.imshow("USB Camera", frame) 
                #keyCode = cv2.waitKey(30) 
                #if keyCode == ord('q'): 
                #    exit() 
                    
            #cv2.destroyAllWindows()

def main():
    object_detector = ObjectDetection()
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()

if __name__ == '__main__':
    main()

