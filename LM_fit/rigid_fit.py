import igl
import trimesh
import numpy as np
import pymesh
import sys
import os
import copy
#from mayavi import mlab
from functools import reduce
from scipy import sparse
import h5py
from scipy.spatial.transform import Rotation
import multiprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from util_package.util import load_PC, parse, angle2rotmatrix, angle2drotmatrix,\
    init_reg, normalize

from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point

def apply_transform(S, T, R, tet_v):
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr

def points2mesh_label(tran_v, meshes, point_cloud, PC_label, i):
    mesh_FF = trimesh.Trimesh(vertices=tran_v, faces=meshes[0].faces)
    mesh_LR = trimesh.Trimesh(vertices=tran_v, faces=meshes[1].faces)
    mesh_RR = trimesh.Trimesh(vertices=tran_v, faces=meshes[2].faces)
    mesh_Front = trimesh.Trimesh(vertices=tran_v, faces=meshes[3].faces)

    points_FF = point_cloud[np.where(PC_label==2), :][0]
    closest_pFF, distance, face_id = trimesh.proximity.closest_point_naive(mesh_FF, points_FF)
    closest_nFF = - mesh_FF.face_normals[face_id,:]
    points_LR = point_cloud[np.where(PC_label==3), :][0]
    closest_pLR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_LR, points_LR)
    closest_nLR = - mesh_LR.face_normals[face_id,:]
    points_RR = point_cloud[np.where(PC_label==4), :][0]
    closest_pRR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_RR, points_RR)
    closest_nRR = - mesh_RR.face_normals[face_id,:]
    points_SR = point_cloud[np.where(PC_label==1), :][0]
    closest_pSR, distance, face_id = trimesh.proximity.closest_point_naive(mesh_Front, points_SR)
    closest_nSR = - mesh_Front.face_normals[face_id,:]
    closest_p = np.concatenate([closest_pFF, closest_pLR, closest_pRR, closest_pSR], axis=0)
    closest_n = np.concatenate([closest_nFF, closest_nLR, closest_nRR, closest_nSR], axis=0)

    #plot_mesh_points(tran_v, np.concatenate([meshes[0].faces, meshes[1].faces, meshes[2].faces, meshes[3].faces]), closest_p)
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

def put__queue(queue, tran_v, meshes, points, PC_label, finished):
    points_FF = points[np.where(PC_label == 0), :][0]
    points_LR = points[np.where(PC_label == 1), :][0]
    points_RR = points[np.where(PC_label == 2), :][0]
    points_SR = points[np.where(PC_label == 3), :][0]
    queue[0].put(points_FF)
    queue[1].put(points_LR)
    queue[2].put(points_RR)
    queue[3].put(points_SR)
    queue[4].put(tran_v)
    queue[5].put(finished)

def LM_rigid(tet_v, tet_f, point_cloud, meshes, PC_lable, queue, liver_model, index):
    #Sinit, Tinit, Rinit = init_reg(point_cloud, PC_lable, tet_v, meshes)
    #Rinit = Rotation.from_matrix(Rinit).as_rotvec()
    #t = Rinit[1]
    #Rinit[1] = Rinit[2]
    #Rinit[2] = t
    S = np.random.uniform(0.9, 1.1, size=1)
    T = 0.0*np.random.normal(size=[3])
    R = np.random.uniform(0, 6.28, size=3)
    #R = np.asarray([3.05, 0.12, 5.36])
    S, T, R = Iterative_Closest_Point.ICP(liver_model, point_cloud, PC_lable, (0, 1))
    parameter = np.concatenate([S, T, R], axis=0)

    min = 10.0
    for i in range(80):
        print("************"+str(index)+'****'+str(i)+"**************")
        #update registration
        S, T, R = parse(parameter)
        tran_v, rotation_matrix = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)
        #if (i == 0):
        #    scene_objs = plot_mesh_points_label(tran_v, meshes, point_cloud, PC_lable)
        #    mlab.show()
        #else:
        #    update_scene_objs(scene_objs, tran_v, meshes, point_cloud, PC_lable)
        finished = (i == 79)
        #put_queue(queue, tran_v, meshes, point_cloud, PC_lable, finished)
        #put_para(parameter)
        closest_points, closest_face_normal = points2mesh_label(tran_v, meshes, point_cloud, PC_lable, i)

        difference = point_cloud - closest_points
        #
        normalized_difference = normalize(difference)
        closest_face_normal[np.where(PC_lable>=2)[0], :] = normalized_difference[np.where(PC_lable>=2)[0], :]
        #
        residual = (1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal * difference, axis=1)
        criteria = np.abs(np.mean(residual))
        if min > criteria:
            min = criteria
            para = parameter
        #if criteria > 0.002:
        #    return 0, 0

        #if i%78==0 :
        #    plot_mesh_points(tran_v, tet_f, point_cloud)
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
        jacobian = np.concatenate([drdS, drdT, drdR3, drdR2, drdR1], axis=1)

        #update parameter
        jTj = np.matmul(jacobian.T, jacobian)
        increament = - np.matmul(np.matmul(np.linalg.inv(jTj + 0.5*np.diag(jTj)), jacobian.T), residual)
        #print(increament)
        print(parameter)
        parameter = parameter + 0.05*increament
    return min, para

def get_queue():
    queue = [multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue()]
    return queue

def one_model(tet_v, tet_f, meshes, index, liver_model):
    save_index(str(index).zfill(3))
    point_cloud, PC_label = load_PC('../../org/datasets/Set' + str(index).zfill(3))
    point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = point_cloud / 1000

    min = 0
    #while min==0:
    queue = get_queue()
    #process = multiprocessing.Process(target=LM_rigid,
    #                                  args=(tet_v, tet_f, point_cloud, meshes, PC_label, queue))
    min, parameter = LM_rigid(tet_v, tet_f, point_cloud, meshes, PC_label, queue, liver_model, index)
    #mlab.figure()
    '''r=animation(queue, meshes)
    process.start()
    #mlab.show()
    process.join()
    min = 1'''
    with h5py.File('../../org/datasets/Set' + str(index).zfill(3)+'/rigid.h5', 'w') as f:
        f.create_dataset('error', data=min)
        f.create_dataset('parameters', data=parameter)
        f.close()
    return 0, 0

'''@mlab.animate(delay=200, ui=False)
def animation(queue, meshes):
    is_first = True
    while 1:
        try:
            finished = queue[-1].get()
        except:
            continue
        if finished:
            break
        if is_first:
            is_first = False
            points_FF = queue[0].get()
            points_LR = queue[1].get()
            points_RR = queue[2].get()
            points_SR = queue[3].get()
            mesh_v = queue[4].get()
            visual_obj = plot_mesh_points_label_anim(mesh_v, meshes, points_FF, points_LR, points_RR, points_SR)
        else:
            points_FF = queue[0].get()
            points_LR = queue[1].get()
            points_RR = queue[2].get()
            points_SR = queue[3].get()
            mesh_v = queue[4].get()
            update_scene_objs_anim(visual_obj, mesh_v, points_FF, points_LR, points_RR, points_SR)
        yield
    mlab.close()'''

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
    point_cloud, PC_label = load_PC('../../org/datasets/Set001')

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
    #plot_mesh_points(tet_v, tet_f, point_cloud)

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

    liver_model = LiverModel()
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==1)[0], :])]

    #registration
    min_all = []
    paras = []
    for i in range(112):
        min, para = one_model(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces, meshes, i+1, liver_model)
        min_all.append(min)
        paras.append(np.expand_dims(para, axis=0))
    paras = np.concatenate(paras, axis=0)
    f = h5py.File('results', 'w')
    f.create_dataset('paras', data=paras)
    f.close()
    # finished
    print(min_all)
    print("Done")

def plot_reg():
    from plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, \
        put_queue, plot_mesh_points_label_anim, update_scene_objs_anim
    #registration
    min_all = []
    paras = []
    for i in range(112):
        with h5py.File('../../org/datasets/Set' + str(i+1).zfill(3) + '/rigid.h5', 'r') as f:
            error = f['error'].value
            parameters = f['parameters'][:]
            f.close()
        min_all.append(error)
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R = parse(parameters)
        print(parameters)
        liver_model = LiverModel()
        tran_v, rotation_matrix = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)
        plot_mesh_points(tran_v, liver_model.tetr_mesh.faces, point_cloud)

    print('done')


if __name__ == "__main__":
    plot_reg()