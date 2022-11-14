import os.path
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import open3d as o3d
from open3d.cuda.pybind.visualization import ViewControl

from utils import uint16_MAX
from utils.geometry_utils import sph2cart



def image2sph(im):
    #np.rad2deg(az * 2) + 360, 20 - np.rad2deg(el * 2)
    iter = np.nditer(im, flags=['multi_index'], order='C')
    for rho in iter:
        idx_x = iter.multi_index[0]
        idx_y = iter.multi_index[1]
        az = np.deg2rad(idx_y - 360) / 2.0
        el = np.deg2rad(20 - idx_x) / 2.0
        yield float(rho) / uint16_MAX * 40, az, el

def create_confusion_matrix(pred, truth):
    tp = 0
    tn = 0
    fp = 0
    fn = 0



def main(folder):
    range_dir = Path(os.path.join(folder, "range"))
    mask_dir = Path(os.path.join(folder, "mask"))
    preds_dir = Path(os.path.join(folder, "preds"))



    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for fname in sorted(os.listdir(range_dir)):
        range_image = Image.open(os.path.join(range_dir, fname))
        true_mask = Image.open(os.path.join(mask_dir, fname))
        pred_mask = Image.open(os.path.join(preds_dir, fname))

        true_mask_flat = np.array(true_mask).flatten()
        pred_mask_flat = np.array(pred_mask).flatten()

        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('true_class', bool), ('pred_class', bool)]
        point_cloud = []
        vec_full = o3d.utility.Vector3dVector()
        vec_preds = o3d.utility.Vector3dVector()
        vec_true = o3d.utility.Vector3dVector()
        for i, (rho, az, el) in enumerate(image2sph(range_image)):
            if rho > 0:
                x, y, z = sph2cart(rho, az, el)
                point_cloud.append((x, y, z, true_mask_flat[i] != 0, pred_mask_flat[i] != 0))
                vec_full.append((x, y, z))
                if true_mask_flat[i] != 0:
                    vec_true.append((x, y, z))
                if pred_mask_flat[i] != 0:
                    vec_preds.append((x, y, z))
        pc_arr = np.array(point_cloud, dtype=dtype)


        pcd_full = o3d.geometry.PointCloud(vec_full)
        pcd_preds = o3d.geometry.PointCloud(vec_preds)
        #pcd_preds ,_ = pcd_preds.remove_radius_outlier(nb_points=15, radius=0.2)
        pcd_true = o3d.geometry.PointCloud(vec_true)
        pcd_true ,_ = pcd_true.remove_radius_outlier(nb_points=3, radius=0.1)


        labels = np.array(pcd_preds.cluster_dbscan(eps=0.2, min_points=15, print_progress=True))
        pcd_human = pcd_preds.select_by_index(np.arange(0, len(labels))[labels == 0])
        pcd_human.paint_uniform_color([0, 1, 0])
        max_label = labels.max(initial=0)
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_preds.colors = o3d.utility.Vector3dVector(colors[:, :3])

        print(pcd_human.get_center())
        print(pcd_true.get_center())

        pcd_true.paint_uniform_color([1, 0, 0])
        #pcd_preds.paint_uniform_color([0, 1, 0])

        pcd_full.paint_uniform_color([0.1, 0.1, 0.1])

        #vis.add_geometry(pcd_full)
        vis.add_geometry(pcd_true)
        vis.add_geometry(pcd_human)
        vis.get_view_control().scale(2)
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
    #vis.destroy_window()


if __name__ == '__main__':
    main("/home/alec/Documents/UofT/AER1515/coat_pass3")