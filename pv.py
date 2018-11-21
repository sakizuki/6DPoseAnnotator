# 点群ビューワ

import numpy as np
from open3d import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='point cloud viewer')
    parser.add_argument('--input', nargs='*', help='input data (.ply or .pcd)')
    args = parser.parse_args()
    print(args.input)
    print(len(args.input))


    pcd = []
    for name in args.input:
        tmp = PointCloud()
        tmp = read_point_cloud( name ) 
        pcd.append(tmp)
        print(np.asarray(tmp.points))
        draw_geometries([tmp])

    if len(args.input) == 1:
        draw_geometries([pcd[0]])
    elif len(args.input) == 2:
        draw_geometries([pcd[0],pcd[1]])
    elif len(args.input) == 3:
        draw_geometries([pcd[0],pcd[1],pcd[2]])
    elif len(args.input) == 4:
        draw_geometries([pcd[0],pcd[1],pcd[2],pcd[3]])
    elif 4 < len(args.input):
        print('Too many inputs.')
    