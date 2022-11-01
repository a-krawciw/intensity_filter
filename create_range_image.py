import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured

from PIL import Image
import os

from sensor_msgs_py.point_cloud2 import read_points

from utils import uint16_MAX
from utils.bag_file_parsing import BagFileParser
from utils.geometry_utils import cart2sph


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

        range_image = np.round( uint16_MAX / 40 * range_image)
        range_image[range_image > uint16_MAX] = uint16_MAX
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
