import argparse
import numpy as np
import open3d as o3d
from os.path import join
from os.path import isfile
from os.path import isdir
import os

parser = argparse.ArgumentParser()
parser.add_argument('--d', required=True)
parser.add_argument('--e', required=False)
parser.add_argument('--v', required=False)
args = parser.parse_args()

box = o3d.io.read_triangle_mesh(join('fetched', 'box.ply'))
line = o3d.io.read_line_set(join('fetched', 'line.ply'))


files = os.listdir(join('fetched'))
empty_addr = [f for f in files if isfile(join('fetched', f)) == True and 'empty' in f][0]
empty = o3d.io.read_point_cloud(join('fetched', empty_addr))
pcd = o3d.io.read_point_cloud(args.d)
if args.v is not None:
    pcd = pcd.voxel_down_sample(float(args.v))

if(args.e == "empty"):
    o3d.visualization.draw_geometries([empty, pcd])
elif(args.e == "all"):
    o3d.visualization.draw_geometries([empty, box, line, pcd])
elif(args.e == "box"):
    o3d.visualization.draw_geometries([box, line, pcd])
else:
    o3d.visualization.draw_geometries([pcd])