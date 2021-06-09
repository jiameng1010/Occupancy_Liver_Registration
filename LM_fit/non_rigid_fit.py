import igl
import trimesh
import numpy as np
import copy
from mayavi import mlab
from functools import reduce
from scipy import sparse
import h5py
from scipy.spatial.transform import Rotation

num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

from util import load_PC, parse_non_rigid, angle2rotmatrix, angle2drotmatrix, init_reg, \
    load_disp_solutions
from plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label

def apply_transform_non_rigid(S, T, R, C, tet_v, templates):
    #apply deformation
    tet_v = tet_v + np.reshape(np.matmul(templates, C), [-1, 3])
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr



def points2mesh_label(tran_v, meshes, point_cloud, PC_label):
    mesh_FF = trimesh.Trimesh(vertices=tran_v, faces=meshes[0].faces)
    mesh_LR = trimesh.Trimesh(vertices=tran_v, faces=meshes[1].faces)
    mesh_RR = trimesh.Trimesh(vertices=tran_v, faces=meshes[2].faces)
    mesh_Front = trimesh.Trimesh(vertices=tran_v, faces=meshes[3].faces)

    points_FF = point_cloud[np.where(PC_label==0), :][0]
    closest_pFF, distance, face_id = trimesh.proximity.closest_point_naive(mesh_FF, points_FF)
    closest_nFF = - mesh_FF.face_normals[face_id,:]
    points_LR = point_cloud[np.where(PC_label==1), :][0]
    closest_pLR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_LR, points_LR)
    closest_nLR = - mesh_LR.face_normals[face_id,:]
    points_RR = point_cloud[np.where(PC_label==2), :][0]
    closest_pRR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_RR, points_RR)
    closest_nRR = - mesh_RR.face_normals[face_id,:]
    points_SR = point_cloud[np.where(PC_label==3), :][0]
    closest_pSR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_Front, points_SR)
    closest_nSR = - mesh_Front.face_normals[face_id,:]
    closest_p = np.concatenate([closest_pFF, closest_pLR, closest_pRR, closest_pSR], axis=0)
    closest_n = np.concatenate([closest_nFF, closest_nLR, closest_nRR, closest_nSR], axis=0)
    return closest_p, closest_n

def point2mesh(tran_v, tet_f, point_cloud):
    mesh = trimesh.Trimesh(vertices=tran_v, faces=tet_f)
    closest_p, distance, face_id = trimesh.proximity.closest_point_naive(mesh, point_cloud)
    closest_n = - mesh.face_normals[face_id,:]
    return closest_p, closest_n

'''def points3mesh(tran_v, tet_f, point_cloud):
    closest_p 
    closest_p, distance, face_id = trimesh.proximity.closest_point(mesh, point_cloud)
    closest_n = mesh.face_normals[face_id,:]
    return closest_p, closest_n'''

def put_para(para):
    with h5py.File('../animation/data.h5', 'w') as hf:
        hf.create_dataset("para", data=para)
def save_index(index):
    with open('../animation/index', 'w') as tf:
        tf.writelines(index)

def LM_non_rigid(tet_v, tet_f, point_cloud, meshes, PC_lable, templates):
    num_of_order = 3
    #Sinit, Tinit, Rinit = init_reg(point_cloud, PC_lable, tet_v, meshes)
    #Rinit = Rotation.from_matrix(Rinit).as_rotvec()
    #t = Rinit[1]
    #Rinit[1] = Rinit[2]
    #Rinit[2] = t
    S = np.random.uniform(0.9, 1.1, size=1)
    T = 0.0*np.random.normal(size=[3])
    R = np.random.uniform(0, 6.28, size=3)
    C = np.random.normal(scale=0.01, size=np.sum(num_of_combination[:4]))
    #R = np.asarray([3.05, 0.12, 5.36])
    parameter = np.concatenate([S, T, R, C], axis=0)
    min = 10.0
    for i in range(80):
        print("**************************")
        #update registration
        S, T, R, C = parse_non_rigid(parameter)
        tran_v, rotation_matrix = apply_transform_non_rigid(S, T, R, C, tet_v, templates)
        #if (i == 79) and (__name__ == "__main__"):
        #    plot_mesh_points_label(tran_v, meshes, point_cloud, PC_lable)
        closest_points, closest_face_normal = points2mesh_label(tran_v, meshes, point_cloud, PC_lable)
        put_para(parameter)
        difference = point_cloud - closest_points
        residual = (1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * difference, axis=1)
        criteria = np.abs(np.mean(residual))
        if min > criteria:
            min = criteria
            para = parameter
        if criteria > 0.002:
            return 0, 0
        print(np.mean(residual))
        weight = 0.5*(PC_lable==3) + 1.0*(PC_lable!=3)
        residual = weight * residual


        inv_closest_points = np.matmul(rotation_matrix.T, (closest_points - np.expand_dims(T, axis=0)).T).T / S
        r1, r2, r3, rr = angle2rotmatrix(R)
        dr1, dr2, dr3 = angle2drotmatrix(R)
        drdS = (closest_points - np.expand_dims(T, axis=0)) / S
        drdS = -(1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * drdS, axis=1, keepdims=True)
        drdR1 = S * np.matmul(dr1, np.matmul(np.matmul(r2, r3), inv_closest_points.T)).T
        drdR1 = -(1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * drdR1, axis=1, keepdims=True)
        drdR2 = S * np.matmul(r1, np.matmul(np.matmul(dr2, r3), inv_closest_points.T)).T
        drdR2 = -(1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * drdR2, axis=1, keepdims=True)
        drdR3 = S * np.matmul(r1, np.matmul(np.matmul(r2, dr3), inv_closest_points.T)).T
        drdR3 = -(1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * drdR3, axis=1, keepdims=True)
        drdT = -(1.0/np.sqrt(point_cloud.shape[0])) * closest_face_normal
        drdC = np.random.normal(scale=1.0e-6, size=[residual.shape[0], np.sum(num_of_combination[:4])])
        jacobian = np.concatenate([drdS, drdT, drdR3, drdR2, drdR1, drdC], axis=1)

        #update parameter
        jTj = np.matmul(jacobian.T, jacobian)
        increament = - np.matmul(np.matmul(np.linalg.inv(jTj + 0.0005*np.diag(jTj)), jacobian.T), residual)
        #print(increament)
        #print(parameter)
        parameter = parameter + 0.5*increament
    return min, para

def one_model(tet_v, tet_f, meshes, index, templates):
    save_index(str(index).zfill(3))
    point_cloud, PC_label = load_PC('../../org/datasets/Set' + str(index).zfill(3))
    point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = point_cloud / 1000

    min = 0
    while min==0:
        min, parameter = LM_non_rigid(tet_v, tet_f, point_cloud, meshes, PC_label, templates)
    return min, parameter

def main():
    # load mesh
    '''ff = h5py.File('../../org/Liver_less_tetr_front.hdf5', 'r')
    tet_v = ff['vertices'][:]
    tet_v = tet_v - np.mean(tet_v, axis=0, keepdims=True)
    tet_v = tet_v
    tet_f = ff['faces'][:]
    #tet_t = ff['voxels'][:]
    ff.close()'''
    mesh_liver = trimesh.load('../../org/Liver.off')
    tet_v = np.asarray(mesh_liver.vertices)
    tet_v = tet_v - np.mean(tet_v, axis=0, keepdims=True)
    tet_f = np.asarray(mesh_liver.faces)
    # load Point cloud
    '''point_cloud, PC_label = load_PC('../../org/datasets/Set001')

    point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = point_cloud / 1000
    S = np.asarray([1.0])
    T = np.asarray([0.0, 0.0, 0.0])
    R = np.asarray([0.0, 0.0, 0.0])
    parameter = np.concatenate([S, T, R], axis=0)
    S, T, R = parse(parameter)
    point_cloud, rotation_matrix = apply_transform(S, T, R, point_cloud)
    #plot_points(point_cloud)
    #plot_mesh(tet_v, tet_f)
    #plot_mesh_points(tet_v, tet_f, point_cloud)'''

    #meshes
    mesh_FF = trimesh.load('../../org/Liver_FF.off')
    mesh_LR = trimesh.load('../../org/Liver_LR.off')
    mesh_RR = trimesh.load('../../org/Liver_RR.off')
    mesh_Front = trimesh.load('../../org/Liver_Front.off')
    #mesh_FF = trimesh.Trimesh(vertices=tet_v, faces=mesh_FF.faces)
    #mesh_LR = trimesh.Trimesh(vertices=tet_v, faces=mesh_LR.faces)
    #mesh_RR = trimesh.Trimesh(vertices=tet_v, faces=mesh_RR.faces)
    #mesh_Front = trimesh.Trimesh(vertices=tet_v, faces=mesh_Front.faces)
    meshes = [mesh_FF, mesh_LR, mesh_RR, mesh_Front]

    templates = load_disp_solutions('../displacement_solutions/Y1e4_P04', np.sum(num_of_combination[:4]))

    #registration
    min_all = []
    paras = []
    for i in range(111):
        min, para = one_model(tet_v, tet_f, meshes, i+1, templates)
        min_all.append(min)
        paras.append(np.expand_dims(para, axis=0))
    paras = np.concatenate(paras, axis=0)
    f = h5py.File('results', 'w')
    f.create_dataset('paras', data=paras)
    f.close()
    # finished
    print(min_all)
    print("Done")

if __name__ == "__main__":
    main()