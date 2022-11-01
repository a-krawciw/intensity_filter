import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def sph2cart(r, az, el):
    z = r*np.sin(el)
    hyp = r*np.cos(el)
    y = hyp*np.sin(az)
    x = hyp*np.cos(az)

    return x, y, z

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el


uint16_max = (2 ** 16 - 1)

def image2sph(im):
    #np.rad2deg(az * 2) + 360, 20 - np.rad2deg(el * 2)
    iter = np.nditer(im, flags=['multi_index'], order='C')
    for rho in iter:
        idx_x = iter.multi_index[0]
        idx_y = iter.multi_index[1]
        az = np.deg2rad(idx_y - 360) / 2.0
        el = np.deg2rad(20 - idx_x) / 2.0
        yield float(rho)/uint16_max*40, az, el


def main(folder):
    range_dir = Path(os.path.join(folder, "range"))
    mask_dir = Path(os.path.join(folder, "mask"))
    preds_dir = Path(os.path.join(folder, "preds"))

    plt.ion()

    fig = plt.figure()
    plt.suptitle("Prediction Comparisons")

    ax_true = fig.add_subplot(1, 2, 1)
    ax_pred = fig.add_subplot(1, 2, 2)

    plt.show()

    for fname in sorted(os.listdir(range_dir)):
        range_image = Image.open(os.path.join(range_dir, fname))
        true_mask = Image.open(os.path.join(mask_dir, fname))
        pred_mask = Image.open(os.path.join(preds_dir, fname))

        true_mask_flat = np.array(true_mask).flatten()
        pred_mask_flat = np.array(pred_mask).flatten()

        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('true_class', bool), ('pred_class', bool)]
        point_cloud = []
        for i, (rho, az, el) in enumerate(image2sph(range_image)):
            if rho > 0:
                x, y, z = sph2cart(rho, az, el)
                point_cloud.append((x, y, z, true_mask_flat[i] != 0, pred_mask_flat[i] != 0))
        pc_arr = np.array(point_cloud, dtype=dtype)
        humans = pc_arr[pc_arr['true_class']]
        pred_humans = pc_arr[pc_arr['pred_class']]
        #ax.clear()
        #ax.scatter(pc_arr['x'], pc_arr['y'], pc_arr['z'], c="g", s=0.1, alpha=0.1)
        #ax.scatter(humans['x'], humans['y'], humans['z'], c="r", s=0.5)
        #plt.title(f"Frame {fname}")
        #ax.set_xlim([-10, 10])
        #ax.set_ylim([-10, 10])
        #ax.set_zlim([-10, 2])

        ax_pred.clear()
        ax_true.clear()
        ax_true.scatter(pc_arr['x'], pc_arr['y'], c="g", s=0.1, alpha=0.1)
        ax_true.scatter(humans['x'], humans['y'], c='r', s=0.5)
        ax_pred.scatter(pc_arr['x'], pc_arr['y'], c="g", s=0.1, alpha=0.1)
        ax_pred.scatter(pred_humans['x'], pred_humans['y'], c='r', s=0.5)

        ax_true.axis("equal")
        ax_pred.axis("equal")
        ax_pred.set_title("Model Predictions")
        ax_true.set_title("Ground Truth")

        ax_pred.set_xlim([-10, 10])
        ax_pred.set_ylim([-10, 10])
        ax_true.set_xlim([-10, 10])
        ax_true.set_ylim([-10, 10])

        fig.canvas.draw()
        fig.canvas.flush_events()





    pass

if __name__ == '__main__':
    main("/home/asrlab/Documents/rosbag_lidar/coat_pass3")