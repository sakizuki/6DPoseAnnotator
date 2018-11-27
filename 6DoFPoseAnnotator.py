# 6DoF pose annotator 
# Shuichi Akizuki, Keio Univ.
# Email: akizuki@elec.keio.ac.jp
#
import open3d as o3 
import numpy as np
import cv2
import copy
import argparse
import common3Dfunc as c3D
from math import *

""" Object model to be transformed """
CLOUD_ROT = o3.PointCloud()
""" Total transformation"""
all_transformation = np.identity(4)
""" Step size for rotation """
step = 0.1*pi

def get_argumets():
    """
        Parse arguments from command line
    """

    parser = argparse.ArgumentParser( description='Interactive 6DoF pose annotator')
    parser.add_argument('--cimg', type=str, default='data/rgb.png',
                        help='file name of the RGB image of the input scene.')
    parser.add_argument('--dimg', type=str, default='data/depth.png',
                        help='file name of the depth image of the input scene. We assume that RGB and depth image have pixel-to-pixel correspondence.')
    parser.add_argument('--intrin', type=str, default='data/realsense_intrinsic.json',
                        help='file name of the camera intrinsic.')
    parser.add_argument('--model', type=str, default='data/hammer_1.pcd',
                        help='file name of the object model (.pcd or .ply).')
    
    return parser.parse_args()

class Mapping():
    def __init__(self, camera_intrinsic_name, _w=640, _h=480, _d=1000.0 ):
        self.camera_intrinsic = o3.read_pinhole_camera_intrinsic(camera_intrinsic_name)
        self.width = _w
        self.height = _h
        self.d = _d
        self.camera_intrinsic4x4 = np.identity(4)
        self.camera_intrinsic4x4[0,0] = self.camera_intrinsic.intrinsic_matrix[0,0]
        self.camera_intrinsic4x4[1,1] = self.camera_intrinsic.intrinsic_matrix[1,1]
        self.camera_intrinsic4x4[0,3] = self.camera_intrinsic.intrinsic_matrix[0,2]
        self.camera_intrinsic4x4[1,3] = self.camera_intrinsic.intrinsic_matrix[1,2]
        
    def showCameraIntrinsic(self):
        print(self.camera_intrinsic.intrinsic_matrix)
        print(self.camera_intrinsic4x4)

    def Cloud2Image( self, cloud_in ):
        
        img = np.zeros( [self.height, self.width], dtype=np.uint8 )
        
        cloud_np = np.asarray(cloud_in.points)
        cloud_np = cloud_np[:,:] / cloud_np[:,[2]]

        cloud_min = np.min(cloud_np,axis=0)

        cloud_mapped = o3.PointCloud()
        cloud_mapped.points = o3.Vector3dVector(cloud_np)
        cloud_mapped.transform(self.camera_intrinsic4x4)
        #print(np.asarray(cloud_mapped.points))
        
        for i, pix in enumerate(cloud_mapped.points):
            if pix[0]<self.width and 0<pix[0] and pix[1]<self.height and 0<pix[1]:
               img[int(pix[1]),int(pix[0])] = int(255.0*(cloud_np[i,2]/cloud_min[2]))
        
        return img
    
    def Pix2Pnt( self, pix, val ):
        pnt = np.array([0.0,0.0,0.0], dtype=np.float)
        depth = val / self.d
        #print('[0,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[0,2]))
        #print('[1,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[1,2]))
        #print(self.camera_intrinsic.intrinsic_matrix)
        pnt[0] = (float(pix[0]) - self.camera_intrinsic.intrinsic_matrix[0,2]) * depth / self.camera_intrinsic.intrinsic_matrix[0,0]
        pnt[1] = (float(pix[1]) - self.camera_intrinsic.intrinsic_matrix[1,2]) * depth / self.camera_intrinsic.intrinsic_matrix[1,1]
        pnt[2] = depth

        return pnt


def mouse_event(event, x, y, flags, param):
    w_name, img_c, img_d, mapping = param

    """Direct move. Object model will be moved to clicked position."""
    if event == cv2.EVENT_LBUTTONUP:
        global all_transformation
        print('Clicked({},{}): depth:{}'.format(x, y, img_d[y,x]))
        print(img_d[y,x])
        pnt = mapping.Pix2Pnt( [x,y], img_d[y,x] )
        print('3D position is', pnt)

        #compute current center of the cloud
        cloud_c = copy.deepcopy(CLOUD_ROT)
        cloud_c, center = c3D.Centering(cloud_c)
        np_cloud = np.asarray(cloud_c.points) 

        np_cloud += pnt
        print('Offset:', pnt )
        offset = np.identity(4)
        offset[0:3,3] -= center
        offset[0:3,3] += pnt
        all_transformation = np.dot( all_transformation, offset )

        CLOUD_ROT.points = o3.Vector3dVector(np_cloud)
        img_m = mapping.Cloud2Image( CLOUD_ROT )
        img_m2 = cv2.merge((img_m,img_m,img_m,))
        img_imposed = cv2.addWeighted(img_m2, 0.5,img_c, 0.5, 0 )

        print("Total transformation is\n", all_transformation )
        cv2.imshow( w_name, img_imposed )


# Pose refinement by ICP
def refine_registration(source, target, trans, voxel_size):
    global all_transformation
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3.registration_icp(source, target, 
            distance_threshold, trans,
            o3.TransformationEstimationPointToPoint())
    
    """ update all transformation""" 
    all_transformation = np.dot( all_transformation, result.transformation )

    return result.transformation


def icp_registration(source, target, voxel_size):
    distance_threshold = voxel_size
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    trans = []
    scale = pi/3.0
    for r in range(3):
        roll = scale*r
        for p in range(3):
            pitch = scale*p
            for y in range(3):
                yaw = scale*y
                trans_tmp = c3D.ComputeTransformationMatrixAroundCentroid( source, roll, pitch, yaw )
                trans.append( trans_tmp )

    result_fitness = 999.0
    result = np.identity(4)
    trans_id = 0

    for i, t in enumerate(trans):
    
        cloud_rot = copy.deepcopy(source)

        #cloud_tmp = copy.deepcopy(cloud_rot)
        #o3.write_point_cloud("rot"+str(i)+".ply", cloud_tmp )
        #cloud_tmp.transform(t)
        #o3.write_point_cloud("rot"+str(i)+"_.ply", cloud_tmp )

        result_tmp = o3.registration_icp(cloud_rot, target, 
                distance_threshold, t,
                o3.TransformationEstimationPointToPoint())
        #print('Score: ',result_tmp.inlier_rmse)
        #print('Fitness : ',result_tmp.fitness)
        #if result_fitness < result_tmp.fitness:
        if result_tmp.inlier_rmse < result_fitness and 0.1 < result_tmp.fitness:
            result_fitness = result_tmp.inlier_rmse
            result = result_tmp.transformation
            trans_id = i
            cloud_rot.transform( result_tmp.transformation )
            o3.write_point_cloud("icp_result.ply", cloud_rot)
            print(" Update: ", result_fitness)
            print(' Fitness : ',result_tmp.fitness)
            print(result)
            print(trans[trans_id])

    print("trans id: ", trans_id)

    return result, result_fitness

def generateImage( mapping, im_color ):
    global CLOUD_ROT

    img_m = mapping.Cloud2Image( CLOUD_ROT )
    img_m2 = cv2.merge((img_m,img_m,img_m,))
    img_mapped = cv2.addWeighted(img_m2, 0.5, im_color, 0.5, 0 )
    cv2.imshow( window_name, img_mapped )



if __name__ == "__main__":

    args = get_argumets()

    """Data loading"""
    print(":: Load two point clouds to be matched.")
    color_raw = o3.read_image( args.cimg )
    depth_raw = o3.read_image( args.dimg )
    camera_intrinsic = o3.read_pinhole_camera_intrinsic( args.intrin )

    im_color = np.asarray(color_raw)
    im_color = cv2.cvtColor( im_color, cv2.COLOR_BGR2RGB )
    im_depth = np.asarray(depth_raw)

    rgbd_image = o3.create_rgbd_image_from_color_and_depth( color_raw, depth_raw, 1000.0, 2.0 )
    pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, camera_intrinsic )
    cloud_in_ds = o3.voxel_down_sample(pcd, 0.005)
    o3.write_point_cloud( "cloud_in_ds.ply", cloud_in_ds )

    np_pcd = np.asarray(pcd.points)

    """Loading and rescaling of the object model"""
    print('Loading: {}'.format(args.model))
    cloud_m = o3.read_point_cloud( args.model )
    cloud_m_ds = o3.voxel_down_sample(cloud_m, 5.0)
    cloud_m_c, _ = c3D.Centering(cloud_m_ds)
    cloud_m_c = c3D.Scaling(cloud_m_c, 0.001)

    CLOUD_ROT = copy.deepcopy(cloud_m_c)

    mapping = Mapping('./data/realsense_intrinsic.json')
    img_mapped = mapping.Cloud2Image( CLOUD_ROT )

    """Mouse event"""
    window_name = '6DoF Pose Annotator'
    cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback( window_name, mouse_event, 
                        [window_name, im_color, im_depth, mapping])

    score = 999.9
    cv2.imshow( window_name, img_mapped )
    while (True):
        key = cv2.waitKey(1) & 0xFF
        """Quit"""
        if key == ord("q"):
            break
        """ICP registration"""
        if key == ord("i"):
            print('ICP start')
            voxel_size = 0.005
            result_icp = refine_registration( CLOUD_ROT, pcd, np.identity(4), 10*voxel_size)
            #result_icp, score_tmp = icp_registration( CLOUD_ROT, cloud_in_ds, voxel_size)
            #print(result_icp)
            #print("score:{}, current score:{}".format(score, score_tmp))
            #if score_tmp < score:
            CLOUD_ROT.transform( result_icp )

            generateImage( mapping, im_color )

        """Step rotation"""
        if key == ord("1"):
            print('Rotation around roll axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, step, 0, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( all_transformation, rotation )

            generateImage( mapping, im_color )
            
        if key == ord("2"):
            print('Rotation around pitch axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, step, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( all_transformation, rotation )

            generateImage( mapping, im_color )
            
        if key == ord("3"):
            print('Rotation around yaw axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, 0, step )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( all_transformation, rotation )

            generateImage( mapping, im_color )
            

    cv2.destroyAllWindows()

    o3.write_point_cloud( "cloud_m.ply", CLOUD_ROT )