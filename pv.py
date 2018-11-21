# 点群ビューワ

import numpy as np
from open3d import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="point cloud viewer")
    parser.add_argument("input", help="input data (.ply or .pcd)")
    args = parser.parse_args()
    print(args.input)
    pcd = read_point_cloud(args.input)
    print(pcd)
    print(np.asarray(pcd.points))
    draw_geometries([pcd])