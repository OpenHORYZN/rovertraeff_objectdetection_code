import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
cap = cv2.VideoCapture(0) # or simulated camera topic
9
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
    parameters=parameters)
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detected marker IDs:", ids)
    cv2.imshow('Aruco Detection', frame)
    if cv2.waitKey(1) == 27: # ESC to exit
        break

rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners,
markerLength=0.15,
cameraMatrix=K,
distCoeffs=D)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.pub = self.create_publisher(Point, '/target_position', 10)
        self.create_subscription(Image, '/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image', self.image_cb, 10)
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        self.params = aruco.DetectorParameters_create()
        self.marker_length = 0.48

    def image_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is not None:
            print("Detected marker IDs:", ids)

            #<horizontal_fov>1.74</horizontal_fov>
            #  <image>
            #    <width>1280</width>
            #    <height>960</height>
            K = np.array([[540, 0, 640], 
                          [0, 542, 480], 
                          [0, 0,   1]], dtype=np.float32)
            D = np.zeros((4,1), dtype=np.float32)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, cameraMatrix=K, distCoeffs=D)

            tvec = tvecs[0][0]
            target = Point()
            target.x = float(tvec[0])
            target.y = float(tvec[1])
            target.z = float(tvec[2])

            self.pub.publish(target)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()