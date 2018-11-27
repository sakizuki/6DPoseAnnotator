# Common functions for 3D processiong 
# Shuichi Akizuki, Keio Univ.
# Email: akizuki@elec.keio.ac.jp
#
import open3d as o3 
import numpy as np
import copy
from math import *

def Centering( _cloud_in ):
    """
    Centering()
    offset an input cloud to its centroid.
    
    input(s):
        _cloud_in: point cloud to be centered
    output(s):
        cloud_off: 
    """
    cloud_in = copy.deepcopy(_cloud_in)
    np_m = np.asarray(cloud_in.points) 
    center = np.mean(np_m, axis=0)
    np_m[:] -= center
    
    cloud_off = o3.PointCloud()
    cloud_off.points = o3.Vector3dVector(np_m)
    
    return cloud_off, center

def ComputeTransformationMatrixAroundCentroid( _cloud_in, _roll, _pitch, _yaw ):

    """ offset center """
    np_in = np.asarray(_cloud_in.points) 
    center = np.mean(np_in, axis=0)
    offset = np.identity(4)
    offset[0:3,3] -= center

    """ rotation """
    rot = RPY2Matrix4x4( _roll, _pitch, _yaw )
    
    """ reverse offset """
    reverse = np.identity(4)
    reverse[0:3,3] = center

    final = np.dot( reverse, np.dot( rot, offset ) )

    return final 


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

def Offset( cloud_in, offset ):
    cloud_np = np.asarray(cloud_in.points)
    cloud_np += offset
    cloud_off = o3.PointCloud()
    cloud_off.points = o3.Vector3dVector(cloud_np)
    return cloud_off

def RPY2Matrix4x4( roll, pitch, yaw ):

    rot = np.identity(4)
    if roll< -3.141:
        roll += 6.282
    elif 3.141 < roll:
        roll -= 6.282
    if pitch < -3.141:
        pitch += 6.282
    elif 3.141 < pitch:
        pitch -= 6.282
    if yaw < -3.141: 
        yaw += 6.282
    elif 3.141 < yaw:
        yaw -= 6.282
    
    rot[ 0, 0 ] = cos(yaw)*cos(pitch)
    rot[ 0, 1 ] = -sin(yaw)*cos(roll) + (cos(yaw)*sin(pitch)*sin(roll))
    rot[ 0, 2 ] = sin(yaw)*sin(roll) + (cos(yaw)*sin(pitch)*cos(roll))
    rot[ 1, 0 ] = sin(yaw)*cos(pitch)
    rot[ 1, 1 ] = cos(yaw)*cos(roll) + (sin(yaw)*sin(pitch)*sin(roll))
    rot[ 1, 2 ] = -cos(yaw)*sin(roll) + (sin(yaw)*sin(pitch)*cos(roll))
    rot[ 2, 0 ] = -sin(pitch)
    rot[ 2, 1 ] = cos(pitch)*sin(roll)
    rot[ 2, 2 ] = cos(pitch)*cos(roll)
    rot[ 3, 0 ] = rot[ 3, 1 ] = rot[ 3, 2 ] = 0.0
    rot[ 3, 3 ] = 1.0

    return rot

def Mat2RPY( rot ):
    roll = atan2( rot[2, 1], rot[2, 2] )
    pitch = atan2(-rot[2, 0], sqrt( rot[2, 1]*rot[2, 1]+rot[2, 2]*rot[2, 2] ) )
    yaw = atan2( rot[1, 0], rot[0, 0] )
    
    return roll, pitch, yaw

def makeTranslation4x4( offset ):

    trans = np.identity(4)
    trans[0:3,3] = offset

    return trans
