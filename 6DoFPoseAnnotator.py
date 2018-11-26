# 6DoF pose annotator 
# Shuichi Akizuki, Keio Univ.
# Email: akizuki@elec.keio.ac.jp
#
import open3d as o3 
import numpy as np
import cv2
import copy
import common3Dfunc as c3D
from math import *

cloud_rot = o3.PointCloud()

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
        #cv2.circle(img, (x, y), 50, (0, 0, 255), -1)
        print('Clicked({},{}): depth:{}'.format(x, y, img_d[y,x]))
        print(img_d[y,x])
        pnt = mapping.Pix2Pnt( [x,y], img_d[y,x] )
        print('3D position is', pnt)

        #compute current center of the cloud
        cloud_c = copy.deepcopy(cloud_rot)
        cloud_c, _ = c3D.Centering(cloud_c)
        np_cloud = np.asarray(cloud_c.points) 

        np_cloud += pnt
        print('Offset:', pnt )

        cloud_rot.points = o3.Vector3dVector(np_cloud)
        img_m = mapping.Cloud2Image( cloud_rot )
        img_m2 = cv2.merge((img_m,img_m,img_m,))
        img_imposed = cv2.addWeighted(img_m2, 0.5,img_c, 0.5, 0 )

        cv2.imshow( w_name, img_imposed )
        o3.write_point_cloud( "cloud_move.pcd", cloud_rot )
    # 右クリック + Shiftキーで緑色のテキストを生成
    #elif event == cv2.EVENT_RBUTTONUP and flags & cv2.EVENT_FLAG_SHIFTKEY:
    #    cv2.putText(img, "CLICK!!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3, cv2.CV_AA)
    
    # 右クリックのみで青い四角形を生成
    #elif event == cv2.EVENT_RBUTTONUP:
    #    cv2.rectangle(img, (x-100, y-100), (x+100, y+100), (255, 0, 0), -1)



#sourceをtransformationによって剛体変換してtargetと一緒に表示
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3.draw_geometries([source_temp, target_temp])


#ICPによるリファイン
def refine_registration(source, target, trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3.registration_icp(source, target, 
            distance_threshold, trans,
            o3.TransformationEstimationPointToPoint())
    return result


def icp_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    trans = []
    trans.append( c3D.RPY2Matrix4x4( 0, 0, 0 ) )
    trans.append( c3D.RPY2Matrix4x4( pi, 0, 0 ) )
    trans.append( c3D.RPY2Matrix4x4( 0, pi, 0 ) )
    trans.append( c3D.RPY2Matrix4x4( 0, 0, pi ) )

    result_score = 999.9
    result = np.identity(4)
    for t in trans:
    
        cloud_rot, offset  = c3D.Centering( source )
        cloud_rot.transform( t )
        cloud_rot = c3D.Offset( cloud_rot, offset )

        result_tmp = o3.registration_icp(cloud_rot, target, 
                distance_threshold, t,
                o3.TransformationEstimationPointToPoint())
        if result_tmp.inlier_rmse < result_score:
            result_score = result_tmp.inlier_rmse
            result = result_tmp.transformation
            print(result_score)

    return result

if __name__ == "__main__":
    """Data loading"""
    print(":: Load two point clouds to be matched.")
    color_raw = o3.read_image("./data/rgb.png")
    depth_raw = o3.read_image("./data/depth.png")
    camera_intrinsic = o3.read_pinhole_camera_intrinsic("./data/realsense_intrinsic.json")

    im_color = np.asarray(color_raw)
    im_depth = np.asarray(depth_raw)

    rgbd_image = o3.create_rgbd_image_from_color_and_depth( color_raw, depth_raw, 1000.0, 2.0 )
    pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, camera_intrinsic )
    o3.write_point_cloud( "cloud_in.ply", pcd )

    np_pcd = np.asarray(pcd.points)

    # 物体モデルの読み込みと大きさ修正
    model_name = "./data/hammer_1.pcd"
    print('Loading: {}'.format(model_name))
    cloud_m = o3.read_point_cloud( model_name )
    cloud_m_ds = o3.voxel_down_sample(cloud_m, 5.0)
    cloud_m_c, _ = c3D.Centering(cloud_m_ds)
    cloud_m_c = c3D.Scaling(cloud_m_c, 0.001)

    cloud_rot = copy.deepcopy(cloud_m_c)

    #offset = np.array([0.03,0.13,0.88])
    #np_tmp = np.asarray(cloud_m_c.points) 
    #np_tmp += offset
    #cloud_m_c.points = o3.Vector3dVector(np_tmp)

    #print(cloud_m_c)
    #print(np.asarray(cloud_m_c.points))

    mapping = Mapping('./data/realsense_intrinsic.json')
    img_mapped = mapping.Cloud2Image( cloud_rot )

    """Mouse event"""
    window_name = '6DoF Pose Annotator'
    cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback( window_name, mouse_event, 
                        [window_name, im_color, im_depth, mapping])


    cv2.imshow( window_name, img_mapped )
    while (True):
        key = cv2.waitKey(1) & 0xFF
        """Quit"""
        if key == ord("q"):
            break
        """ICP registration"""
        if key == ord("i"):
            print('ICP start')
            voxel_size = 0.02
            #result_icp = refine_registration( cloud_rot, pcd, np.identity(4), 10*voxel_size)
            result_icp = icp_registration( cloud_rot, pcd, 10*voxel_size)
            print(result_icp)
            cloud_rot.transform( result_icp )

            img_m = mapping.Cloud2Image( cloud_rot )
            img_m2 = cv2.merge((img_m,img_m,img_m,))
            img_mapped = cv2.addWeighted(img_m2, 0.5, im_color, 0.5, 0 )

            cv2.imshow( window_name, img_mapped )
            

    cv2.destroyAllWindows()

    o3.write_point_cloud( "cloud_m.ply", cloud_rot )

    np_depth = np.asarray(depth_raw)
    print('max_depth:', np.max(np_depth))
    print('min_depth:', np.min(np_depth))