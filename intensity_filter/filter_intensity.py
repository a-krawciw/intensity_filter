from matplotlib import pyplot as plt
import rclpy
from rclpy.node import Node

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from sklearn.cluster import AffinityPropagation, KMeans


from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points

def euclidean_3d(points, query):
    return np.sqrt((points['x']-query['x'])**2 + (points['y'] - query['y'])**2 + (points['z'] - query['z'])**2)

def filter_by_radius(point_cloud, radius, dist_func=None):
    if dist_func is None:
        dist_func = euclidean_3d

    mask_shape = len(point_cloud)
    output_mask = np.full(mask_shape, False)
    for i, point in enumerate(point_cloud):
        mask = np.full(mask_shape, True)
        mask[i] = False
        dists = dist_func(point_cloud[mask], point)
        output_mask[i] = np.min(dists) > radius

    return np.delete(point_cloud, output_mask)

def find_duplication_pattern(point_cloud):

    points_arr = structured_to_unstructured(point_cloud[['x', 'y', 'z']])
    sorted_data = points_arr[np.lexsort(points_arr.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    return point_cloud[row_mask]

PLOT = False

class PointCloudIntensityOffset(Node):

    def __init__(self, fov=70.0):
        super().__init__('FilterCoat')
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.point_cloud_cb,
            10)
        self.publisher = self.create_publisher(PointCloud2, 'points_cropped', 10)
        self.eps = fov/2.0
        self.subscription  # prevent unused variable warning

        if PLOT:

            self.x = []
            self.y = []
            self.t = []
            plt.ion()

            self.figure, self.ax = plt.subplots(figsize=(4,5))
            self.plot1, = self.ax.plot([], [], 'o')

            plt.xlabel("X pos (m)",fontsize=18)
            plt.ylabel("Y pos (m)",fontsize=18)
            plt.xlim([0, 10])
            plt.ylim([0, 10])
            plt.show()

    def point_cloud_cb(self, msg):
        points = read_points(msg)
        points = np.delete(points, (points['intensity'] < 252) | (points['intensity'] > 254))
        points = find_duplication_pattern(points)

        #print(points[:50])
        #points = np.delete(points, [i % 2 for i in range(len(points))])
        #print(points[:50])
        points = filter_by_radius(points, 0.05)
        
        

        #kmeans = KMeans(n_clusters = 4).fit(structured_to_unstructured(points[['x', 'y', 'z']]))
        #points['intensity'] = kmeans.labels_

        #affinity = AffinityPropagation().fit(structured_to_unstructured(points[['x', 'y', 'z']]))
        #points['intensity'] = affinity.labels_

        if PLOT:
            self.x.append(np.mean(points['x']))
            self.y.append(np.mean(points['y']))
            self.t.append(msg.header.stamp)

            self.plot1.set_xdata(self.x)
            self.plot1.set_ydata(self.y)

            
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()


        data = points.tobytes() 
        msg.data = data
        msg.width = points.shape[0]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    point_cloud_remap = PointCloudIntensityOffset()

    rclpy.spin(point_cloud_remap)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    point_cloud_remap.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

