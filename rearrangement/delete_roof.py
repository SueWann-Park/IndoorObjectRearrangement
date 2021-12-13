# make pcd files for every label
import numpy as np
import open3d as o3d
import os
import math

def delete_roof():
    point_cloud = np.loadtxt("./office_1.txt")
    segmentation = np.load("./predictions_office_1.npy", allow_pickle = True)
    NUM_LABEL = segmentation.max()

    saved_without_roof = []
    for i in range(0, len(point_cloud)):
        if(segmentation[i] >= 2):
            saved_without_roof.append(point_cloud[i, :].tolist())
    result_without_roof = np.array(saved_without_roof)
    pcd_without_roof = o3d.geometry.PointCloud()
    pcd_without_roof.points = o3d.utility.Vector3dVector(result_without_roof[:,:3])
    pcd_without_roof.colors = o3d.utility.Vector3dVector(result_without_roof[:,3:6]/255.0)
    o3d.io.write_point_cloud("./original_without_roof.pcd", pcd_without_roof)
    

if( __name__ == '__main__'):
    delete_roof()