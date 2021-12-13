# make pcd files for every label
import numpy as np
import open3d as o3d
import os
import math

def fetch():
    point_cloud = np.loadtxt("./gsr_3.txt")
    segmentation = np.load("./predictions_gsr_3.npy", allow_pickle = True)
    NUM_LABEL = segmentation.max()

    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    origin_pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255.0)
    origin_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    o3d.visualization.draw_geometries([origin_pcd])\
    #o3d.io.write_point_cloud("./original.pcd", origin_pcd)
    
    indices = []
    if not os.path.exists('fetched'):
        os.makedirs('fetched')
    
    boundary = [0,0,0,0,0,0]
    
    for TARGET in range (0, NUM_LABEL+1):
        saved_cloud = []
        for i in range(0, len(point_cloud)):
            if(segmentation[i] == TARGET):
                saved_cloud.append(point_cloud[i, :].tolist())
                if(TARGET >= 8 or TARGET == 6 or TARGET < 2):
                    indices.append(i)

        if len(saved_cloud) == 0:
            continue
        
        result = np.array(saved_cloud)
        
        if(TARGET == 3):
            boundary[0] = result[:, 0].max()
            boundary[1] = result[:, 0].min()
            boundary[2] = result[:, 1].max()
            boundary[3] = result[:, 1].min()
        if(TARGET == 2):
            boundary[5] = result[:, 2].mean()
        if(TARGET == 1):
            boundary[4] = result[:, 2].mean()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(result[:,3:6]/255.0)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=50, print_progress=True))
        
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        #o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud("./fetched/" + str(TARGET) + ".pcd", pcd)
        extract(pcd, TARGET, labels, boundary)
    
    empty_pcd = empty(indices, point_cloud.tolist(), True)
    empty_pcd.translate(np.array([-boundary[1],-boundary[3],0]))
    print("x: " + str(boundary[1]) + ", y: " + str(boundary[3]))
    boundary[0] -= boundary[1]
    boundary[1] = 0
    boundary[2] -= boundary[3]
    boundary[3] = 0
    
    empty_points = np.asarray(empty_pcd.points)
    bounding_box = [[boundary[1],boundary[3],boundary[5]],[boundary[1],boundary[3],boundary[4]],
                    [boundary[1],boundary[2],boundary[5]],[boundary[1],boundary[2],boundary[4]],
                    [boundary[0],boundary[3],boundary[5]],[boundary[0],boundary[3],boundary[4]],
                    [boundary[0],boundary[2],boundary[5]],[boundary[0],boundary[2],boundary[4]]]
    
    line, box = draw_bounding_box(bounding_box)
    #o3d.visualization.draw_geometries([empty_pcd, line_set])
    o3d.io.write_point_cloud("./fetched/empty" + 
        "_" + str(int(boundary[0]*100)) + "_" + str(int(boundary[2]*100)) + "_.pcd", empty_pcd)
    o3d.io.write_line_set("./fetched/line.ply", line)
    o3d.io.write_triangle_mesh("./fetched/box.ply", box)

def empty(indices, empty_scene, isFirst):
    # before: eps=0.08, min_points=50
    # after : eps=0.1,  min_points=100
    print("indices len: " + str(len(indices)) + " scene len: " + str(len(empty_scene)) + " isFirst :" + str(isFirst))
    indices.sort()
    
    temp = np.array(empty_scene)
    empty_scene = []
    i_prev = 0
    for i in indices:
        for j in range(i_prev,i):
            empty_scene.append(temp[j, :].tolist())
        i_prev = i+1
        if i == indices[len(indices)-1]:
            for j in range(i+1, len(temp)):
                empty_scene.append(temp[j, :].tolist())
    
    print("indices len: " + str(len(indices)) + " scene len: " + str(len(empty_scene)) + " isFirst :" + str(isFirst))

    empty_result = np.array(empty_scene)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(empty_result[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(empty_result[:,3:6]/255.0)

    if(isFirst == True):
        new_indices = []
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=100, print_progress=True))
        for i in range (0, len(empty_scene)):
            if(labels[i] == -1):
                new_indices.append(i)
        #o3d.visualization.draw_geometries([pcd])
        return empty(new_indices, empty_scene, False)

    return pcd


def draw_bounding_box(corner_box):
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [0, 2], [1, 3], [4, 6], [5, 7]]
#            [0, 5], [1, 4], [1, 7], [3, 5],
#            [2, 7], [3, 6], [0, 6], [2, 4]]
    triangles = [[2,1,0], [1,2,3], [0,1,4], [5,4,1], [4,5,6], [7,6,5],
                 [6,3,2], [3,6,7], [1,3,5], [7,5,3], [4,2,0], [2,4,6]]
    
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(corner_box)
    line.colors = o3d.utility.Vector3dVector(colors)
    line.lines = o3d.utility.Vector2iVector(lines)
    
    box = o3d.geometry.TriangleMesh()
    box.vertices = o3d.utility.Vector3dVector(corner_box)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    box.compute_vertex_normals()
    return line, box


def extract(pcd, CURRENT, labels, boundary):
    AT_LEAST = 2000
    current_directory = os.path.join("fetched", str(CURRENT))
    if not os.path.exists(current_directory):
        os.makedirs(current_directory)
    
    print("FILE: " + str(CURRENT))
    NUM_OBJECTS = 0

    for x in range (0, labels.max() + 1):
        saved_points = []
        saved_colors = []
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        for i in range(0, len(labels)):
            if (labels[i] == x):
                saved_points.append(points[i, :].tolist())
                saved_colors.append(colors[i, :].tolist())
        
        result_points = np.array(saved_points)
        result_colors = np.array(saved_colors)
        
        x_max = result_points[:, 0].max()
        x_min = result_points[:, 0].min()
        y_max = result_points[:, 1].max()
        y_min = result_points[:, 1].min()
        x_mean = (x_max*100 + x_min*100) / 200.0
        y_mean = (y_max*100 + y_min*100) / 200.0
        
        if(len(result_points) > AT_LEAST):
            print("                           NUM_OBJECTS: " + str(NUM_OBJECTS))
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(result_points[:,:])
            pcd2.colors = o3d.utility.Vector3dVector(result_colors[:,:])
            pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            pcd2.translate(np.array([-x_mean,-y_mean,0]), relative = True)
            
            print("Object: " + str(x_min) + ", " + str(x_max) + ", " + str(y_min) + ", " + str(y_max))
            print("Room  : " + str(boundary[1]) + ", " + str(boundary[0]) + ", " + str(boundary[3]) + ", " + str(boundary[2]))
            position = "O"
            
            width = x_max * 100 - x_min * 100
            height = y_max * 100 - y_min * 100
            
            if(CURRENT == 8 or CURRENT == 11 or CURRENT == 12):
                if( abs(y_max - boundary[2]) * 100 < 20):
                    print("TOP")
                    pcd2 = rotatePCD(pcd2, 90)
                    temp = width
                    width = height
                    height = temp
                elif( abs(y_min - boundary[3]) * 100 < 20):
                    print("BOTTOM")
                    pcd2 = rotatePCD(pcd2, -90)
                    temp = width
                    width = height
                    height = temp
                elif( abs(x_min- boundary[1]) * 100 < 20 ):
                    print("LEFT")
                elif( abs(x_max - boundary[0]) * 100 < 20):
                    print("RIGHT")
                    pcd2 = rotatePCD(pcd2, 180)


                else:
                    position = "I"
            
            elif(CURRENT == 9 or CURRENT == 10):
                position = "I"
            
            o3d.io.write_point_cloud(os.path.join(current_directory,
                position + "_" + str(int(width)) + "_" + str(int(height))
                + "_.pcd"), pcd2)
            NUM_OBJECTS = NUM_OBJECTS +1


def rotatePCD(pcd, angle):
    R = pcd.get_rotation_matrix_from_axis_angle((np.array([0,0, angle * math.pi / 180.0])))
    return pcd.rotate(R, center = (0,0,0))

if( __name__ == '__main__'):
    fetch()