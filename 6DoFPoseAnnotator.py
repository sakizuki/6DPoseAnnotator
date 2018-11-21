# 6DoF pose annotator 
# Shuichi Akizuki, Keio Univ.
# Email: akizuki@elec.keio.ac.jp
#
import open3d as o3 
import numpy as np
import cv2
import copy


def Centering( cloud_in ):
    """
    offset an input cloud to its centroid.
    
    input(s):
        cloud_in: point cloud to be centered
    output(s):
        cloud_off: 
    """
    np_m = np.asarray(cloud_in.points) 
    center = np.mean(np_m, axis=0)
    np_m[:] -= center
    
    cloud_off = o3.PointCloud()
    cloud_off.points = o3.Vector3dVector(np_m)
    
    return cloud_off

def Scaling( cloud_in, scale ):
    """
    multiply scaling factor to the input point cloud.
    input(s):
        cloud_in: point cloud to be scaled.
        scale: scaling factor
    output(s):
        cloud_out: 
    """   
    cloud_np = np.asarray(cloud_in.points) 
    cloud_np *= scale
    cloud_out = o3.PointCloud()
    cloud_out.points = o3.Vector3dVector(cloud_np)
    
    return cloud_out

class Mapping():
    def __init__(self, camera_intrinsic_name, _w=640, _h=480, _d=400.0 ):
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
        print(np.asarray(cloud_mapped.points))
        
        for i, pix in enumerate(cloud_mapped.points):
            if pix[0]<self.width and 0<pix[0] and pix[1]<self.height and 0<pix[1]:
               img[int(pix[1]),int(pix[0])] = int(255.0*(cloud_np[i,2]/cloud_min[2]))
        
        return img

def mouse_event(event, x, y, flags, param):
    w_name, img = param

    # 左クリックで赤い円形を生成
    if event == cv2.EVENT_LBUTTONUP:
        #cv2.circle(img, (x, y), 50, (0, 0, 255), -1)
        print('Clicked({},{})'.format(x, y))
    
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
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3.registration_icp(source, target, 
            distance_threshold, trans,
            o3.TransformationEstimationPointToPoint())
    return result

if __name__ == "__main__":
    #データ読み込み
    print(":: Load two point clouds to be matched.")
    color_raw = o3.read_image("./data/rgb.png")
    depth_raw = o3.read_image("./data/depth.png")
    camera_intrinsic = o3.read_pinhole_camera_intrinsic("./data/realsense_intrinsic.json")


    rgbd_image = o3.create_rgbd_image_from_color_and_depth( color_raw, depth_raw )
    pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, camera_intrinsic)
    o3.write_point_cloud( "cloud_in.pcd", pcd )


    # 物体モデルの読み込みと大きさ修正
    model_name = "./data/hammer_1.pcd"
    print('Loading: {}'.format(model_name))
    cloud_m = o3.read_point_cloud( model_name )
    cloud_m_ds = o3.voxel_down_sample(cloud_m, 2.0)
    cloud_m_c = Centering(cloud_m_ds)
    cloud_m_c = Scaling(cloud_m_c, 0.001)

    print('Down sampling')
    voxel_size = 0.002
    print(np.asarray(cloud_m_c.points))

    offset = np.array([0.03,0.13,0.88])

    np_tmp = np.asarray(cloud_m_c.points) 
    np_tmp += offset
    cloud_m_c.points = o3.Vector3dVector(np_tmp)
    print(cloud_m_c)
    print(np.asarray(cloud_m_c.points))
    o3.write_point_cloud( "cloud_m.pcd", pcd )

    mapping = Mapping('./data/realsense_intrinsic.json')
    img_mapped = mapping.Cloud2Image( cloud_m_c )

    """Mouse event"""
    window_name = '6DoF Pose Annotator'
    cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback( window_name, mouse_event, [window_name, img_mapped])

    while (True):
        cv2.imshow( window_name, img_mapped)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()



    source = cloud_m_c
    target = pcd



    trans = np.identity(4)

    #ICPによる微修正
    result_icp = refine_registration(source, target, trans, 10*voxel_size)
    print(result_icp)
    print('Result transformation is:')
    print(result_icp.transformation)
    draw_registration_result(source, target, result_icp.transformation)
