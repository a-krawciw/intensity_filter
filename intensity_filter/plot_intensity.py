import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points

import matplotlib.pyplot as plt

class PointCloudIntensityOnRay(Node):

    def __init__(self, ray_yaw=0):
        super().__init__('DistanceCompensation')
        self.subscription = self.create_subscription(
            PointCloud2,
            'points',
            self.point_cloud_cb,
            10)
        self.publisher = self.create_publisher(PointCloud2, 'points_scaled', 10)
        self.angle = ray_yaw
        self.eps = 1
        self.subscription  # prevent unused variable warning
        plt.ion()

        self.figure, self.ax = plt.subplots(figsize=(4,5))
        self.plot1, = self.ax.plot([], [], 'o')

        plt.xlabel("Range (m)",fontsize=18)
        plt.ylabel("Intensity",fontsize=18)
        plt.xlim([0, 30])
        plt.ylim([0, 5])
        plt.show()
    

    def point_cloud_cb(self, msg):
        points = read_points(msg)
        x = []
        y = []
        for point in points:
            if abs(point['yaw'] - self.angle) < self.eps:
                x.append(point['range'])
                y.append( point['intensity'])


        self.plot1.set_xdata(x)
        self.plot1.set_ydata(y)

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


        data = points.tobytes() 
        msg.data = data
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    point_cloud_remap = PointCloudIntensityOnRay()

    rclpy.spin(point_cloud_remap)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    point_cloud_remap.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

