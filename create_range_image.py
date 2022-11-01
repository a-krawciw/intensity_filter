import sqlite3

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from PIL import Image
import os

from sensor_msgs_py.point_cloud2 import read_points


class BagFileParser():

    def __init__(self, bag_file):
        try:
            self.conn = sqlite3.connect(bag_file)
        except Exception as e:
            print('Could not connect: ', e)
            raise Exception('could not connect')

        self.cursor = self.conn.cursor()

        ## create a message (id, topic, type) map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
    def get_bag_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]

    def get_bag_msgs_iter(self, topic_name):
        topic_id = self.topic_id[topic_name]
        result = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id))
        while True:
            res = result.fetchone()
            if res is not None:
                yield (res[0], deserialize_message(res[1], self.topic_msg_message[topic_name]))
            else:
                break


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el


def remove_duplicates(point_cloud):
    points_arr = structured_to_unstructured(point_cloud[['x', 'y', 'z']])
    sorted_data = points_arr[np.lexsort(points_arr.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    return point_cloud[row_mask]


def main(filename, out_folder):
    parser = BagFileParser(filename)
    range_image = np.zeros((360 * 2, 80))  # 0.5degree resolution in both axes
    plt.ion()
    figure, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10))
    ims = ax1.imshow(range_image.T, vmax=40)
    bin = ax2.imshow(range_image.T, vmax=1)
    plt.title("Frame 0")
    plt.show()

    if not os.path.exists(os.path.join(out_folder, "range")):
        os.mkdir(os.path.join(out_folder, "range"))
    if not os.path.exists(os.path.join(out_folder, "mask")):
        os.mkdir(os.path.join(out_folder, "mask"))

    for j, (result, msg) in enumerate(parser.get_bag_msgs_iter("/velodyne_points")):
        range_image = np.zeros((360 * 2, 80))  # 0.5degree resolution in both axes
        mask_image = range_image.copy()

        points = read_points(msg)
        print(points.shape)
        points = remove_duplicates(points)
        rho, h_ang, v_ang = cart2sph(points['x'], points['y'], points['z'])
        for i, (r, h_idx, v_idx) in enumerate(zip(rho, np.rad2deg(h_ang * 2) + 360, np.rad2deg(v_ang * 2))):
            range_image[int(h_idx) % 720, 20 - int(v_idx)] = r
            mask_image[int(h_idx) % 720, 20 - int(v_idx)] = 1 if (points['intensity'][i] > 251) & (
                        points['intensity'][i] < 255) else 0


        ims.set_data(range_image.T)

        uint16_max = (2**16 - 1)
        range_image = np.round( uint16_max / 40 * range_image)
        range_image[range_image > uint16_max] = uint16_max
        range_im = Image.fromarray(range_image.T.astype(np.uint16, casting="unsafe"))
        mask_im = Image.fromarray(mask_image.T.astype(np.uint8, casting="unsafe")*255)
        range_im.save(os.path.join(out_folder, f"range/{j:05d}.png"))
        mask_im.save(os.path.join(out_folder, f"mask/{j:05d}.png"))

        print(min(np.rad2deg(v_ang)))
        print(points.shape)

        bin.set_data(mask_image.T)
        figure.canvas.draw()
        figure.canvas.flush_events()


if __name__ == '__main__':
    main("/home/asrlab/Documents/rosbag_lidar/coat_pass3/coat_pass3_0.db3",
         "/home/asrlab/Documents/rosbag_lidar/coat_pass3")
