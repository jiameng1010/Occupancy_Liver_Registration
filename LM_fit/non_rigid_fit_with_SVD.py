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
from util_package.util import load_PC, parse, angle2rotmatrix, angle2drotmatrix, init_reg, load_disp_solutions,\
    parse_non_rigid, normalize, best_fit_transform
from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label,\
    put_queue, plot_mesh_points_label_anim, update_scene_objs_anim, plot_meshes_n_points, plot_meshes

from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point


num_of_combination = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])

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

def apply_transform_non_rigid(S, T, R, C, tet_v, templates):
    #apply deformation
    deformation = np.reshape(np.matmul(templates, C), [-1, 3])
    tet_v = tet_v + deformation
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr, deformation

def apply_transform_non_rigid_reverse(S, T, R, C, tet_v, deformation):
    r1, r2, r3, rr = angle2rotmatrix(R)
    invert_rr = np.linalg.inv(rr)
    tran_v = tet_v - np.expand_dims(T, axis=0)
    tran_v = tran_v / S
    tran_v = np.matmul(invert_rr, tran_v.T).T
    tran_v = tran_v - deformation
    return tran_v, invert_rr, deformation

def points2mesh_label(tran_v, meshes, point_cloud, PC_label, i):
    mesh_FF = trimesh.Trimesh(vertices=tran_v, faces=meshes[0].faces, process=False)
    mesh_LR = trimesh.Trimesh(vertices=tran_v, faces=meshes[1].faces, process=False)
    mesh_RR = trimesh.Trimesh(vertices=tran_v, faces=meshes[2].faces, process=False)
    mesh_Front = trimesh.Trimesh(vertices=tran_v,
                                 faces=np.concatenate([meshes[3].faces, meshes[0].faces, meshes[2].faces, meshes[1].faces],axis=0),
                                 process=False)

    points_FF = point_cloud[np.where(PC_label==2), :][0]
    closest_pFF, distance, face_idFF = trimesh.proximity.closest_point_naive(mesh_FF, points_FF)
    closest_nFF = - mesh_FF.face_normals[face_idFF,:]
    points_LR = point_cloud[np.where(PC_label==3), :][0]
    closest_pLR, distance, face_idLR = trimesh.proximity.closest_point_naive(mesh_LR, points_LR)
    closest_nLR = - mesh_LR.face_normals[face_idLR,:]
    points_RR = point_cloud[np.where(PC_label==4), :][0]
    closest_pRR, distance, face_idRR = trimesh.proximity.closest_point_naive(mesh_RR, points_RR)
    closest_nRR = - mesh_RR.face_normals[face_idRR,:]
    points_SR = point_cloud[np.where(PC_label==1), :][0]
    closest_pSR, distance, face_idSR = trimesh.proximity.closest_point_naive(mesh_Front, points_SR)
    closest_nSR = - mesh_Front.face_normals[face_idSR,:]
    closest_p = np.concatenate([closest_pFF, closest_pLR, closest_pRR, closest_pSR], axis=0)
    closest_n = np.concatenate([closest_nFF, closest_nLR, closest_nRR, closest_nSR], axis=0)
    closest_vid = np.concatenate([mesh_FF.faces[face_idFF],
                                  mesh_LR.faces[face_idLR],
                                  mesh_RR.faces[face_idRR],
                                  mesh_Front.faces[face_idSR]], axis=0)

    #plot_mesh_points(tran_v, np.concatenate([meshes[0].faces, meshes[1].faces, meshes[2].faces, meshes[3].faces]), closest_p)
    return closest_p, closest_n, closest_vid

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

def plot_result(liver_model, point_cloud, tran_v, S, T, R):
    with open('plot_or_not', 'r') as f:
        if f.readline() == 'N\n':
            return
    tran_v_rigid, rotation_matrix_rigid = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)
    plot_meshes_n_points(tran_v, liver_model.faces, \
                         tran_v_rigid, liver_model.faces, \
                         tran_v_rigid, liver_model.faces, \
                         point_cloud)



def LM_rigid(tet_v, tet_f, point_cloud, meshes, PC_lable, queue, liver_model, templates, index):
    alpha = 7e-2
    stiffness = liver_model.get_stiffness()
    S, T, R = Iterative_Closest_Point.ICP(liver_model, point_cloud, PC_lable, (0, 1))
    #S = 1.03
    C = np.zeros(shape=42)
    #C = np.random.normal(0.0, 0.1, C.shape)
    parameter = np.concatenate([S, T, R, C], axis=0)

    min = 10.0
    for i in range(80):
        print("************"+str(index)+'****'+str(i)+"**************")
        #update registration
        S, T, R, C = parse_non_rigid(parameter)
        tran_v, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices, templates)
        #if (i == 0):
        #    scene_objs = plot_mesh_points_label(tran_v, meshes, point_cloud, PC_lable)
        #    mlab.show()
        #else:
        #    update_scene_objs(scene_objs, tran_v, meshes, point_cloud, PC_lable)
        finished = (i == 79)
        #put_queue(queue, tran_v, meshes, point_cloud, PC_lable, finished)
        #put_para(parameter)

        closest_points, closest_face_normal, closest_vid = points2mesh_label(tran_v, meshes, point_cloud, PC_lable, i)
        plot_result(liver_model, point_cloud, tran_v, S, T, R)
        difference = point_cloud - closest_points
        #
        normalized_difference = normalize(difference)
        #closest_face_normal[np.where(PC_lable>=2)[0], :] = normalized_difference[np.where(PC_lable>=2)[0], :]
        closest_face_normal = np.expand_dims(np.sign(np.sum(closest_face_normal * difference, axis=1)), axis=1) * normalized_difference
        #
        residual = (1.0/np.sqrt(point_cloud.shape[0])) * np.sum(closest_face_normal*difference, axis=1)
        E = np.sqrt(alpha) * np.matmul(np.reshape(deformation, [-1]), stiffness.dot(np.reshape(deformation, [-1])))
        E_C = np.sum(C*C)
        residual_all = np.concatenate([residual, np.asarray([E])], axis=0)
        loss1 = np.mean(residual * residual)
        loss2 = E * E
        print('**** ' + str(loss1) + ' **** ' + str(loss2) + ' ****')
        criteria = np.sqrt(np.mean(np.sum(difference*difference, axis=1)))
        if min > criteria:
            min = criteria
            para = parameter
        #if criteria > 0.002:
        #    return 0, 0

        #if i%78==0 :
        #    plot_mesh_points(tran_v, tet_f, point_cloud)
        print(str(criteria) + '  ****  ' + str(E))
        if loss2 > 35:
            return min, para
        #weight = 0.5*(PC_lable==3) + 1.0*(PC_lable!=3)
        #residual = weight * residual

        inv_closest_points = np.matmul(rotation_matrix.T, (closest_points - np.expand_dims(T, axis=0)).T).T / S
        Th,_,_ = best_fit_transform(inv_closest_points, point_cloud)
        R_new = Rotation.from_matrix(Th[:3, :3]).as_euler('xyz')
        T_new = Th[:3, 3]
        nex_rigid = np.concatenate([np.asarray([1.0]), T_new, R_new], axis=0)
        increament_rigid = nex_rigid - parameter[:7]

        contribution2displacement = np.reshape(templates, [-1, 3, 42])
        dDdC = np.mean(contribution2displacement[closest_vid, :, :], axis=1) #mean over 3 vertices of a triangle face
        drdC = S * np.matmul(rotation_matrix, dDdC)
        drdC = (1.0 / np.sqrt(point_cloud.shape[0])) * np.sum(np.expand_dims(closest_face_normal, axis=2) * drdC, axis=1)
        #drdC = - np.concatenate([drdC[:, :4], drdC[:, 4:]], axis=1)
        #drdC =  np.concatenate([drdC[:, :4], drdC[:, 5:]], axis=1)

        dEdD = np.sqrt(alpha) * (stiffness.T).dot(np.reshape(deformation, [-1]))
        dEdc = np.matmul(np.expand_dims(dEdD, axis=0), templates)
        #dEdc = np.concatenate([dEdc[:, :4], dEdc[:, 5:]], axis=1)

        #dE_CdC = np.expand_dims(alpha * C, axis=0)

        jacobian2 = np.concatenate([drdC, dEdc], axis=0)
        #jacobian2 = drdC
        jTj2 = np.matmul(jacobian2.T, jacobian2)

        increament_non_rigid = 0.02 * np.matmul(np.matmul(np.linalg.inv(jTj2 + 0.00*np.diag(jTj2)), jacobian2.T), residual_all)

        #increament_non_rigid = -1e8 * np.mean(jacobian2*np.expand_dims(residual, axis=1), axis=0)
        '''increament = np.concatenate([#np.zeros(shape=[1], dtype=np.float64),
                                     increament_rigid,
                                     increament_non_rigid[:4],
                                     np.zeros(shape=[1], dtype=np.float64),
                                     increament_non_rigid[4:]], axis=0)'''
        if i<20:
            increament = np.concatenate([increament_rigid,
                                         0.3 * increament_non_rigid], axis=0)
        else:
            increament = np.concatenate([0.3*increament_rigid,
                                         increament_non_rigid], axis=0)
        print(parameter)
        parameter = parameter + increament

        '''C_recurrent = parameter[7:]
        for i in range(50):
            deformation = np.reshape(np.matmul(templates, C_recurrent), [-1, 3])
            dEdc = np.matmul(np.expand_dims(dEdD, axis=0), templates)
            dEdD = np.sqrt(alpha) * (stiffness.T).dot(np.reshape(deformation, [-1]))
            C_recurrent -= 1e3 * dEdc[0] * loss2
        parameter[7:] = C_recurrent'''
    return min, para

def get_queue():
    queue = [multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue(),
             multiprocessing.Queue()]
    return queue

def one_model(tet_v, tet_f, meshes, index, liver_model, templates):
    with h5py.File('../../org/datasets/Set' + str(index).zfill(3) + '/nonrigid_without_normal_separate_withoutscale_14p.h5',
                   'r') as f:
        error = f['error'].value
        print(error)
        non_rigid_parameters = f['parameters'][:]
        f.close()
    if error > 0.0015:
        return 0, 0
    save_index(str(index).zfill(3))
    point_cloud, PC_label = load_PC('../../org/datasets/Set' + str(index).zfill(3))
    point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = point_cloud / 1000

    min = 0
    #while min==0:
    queue = get_queue()
    #process = multiprocessing.Process(target=LM_rigid,
    #                                  args=(tet_v, tet_f, point_cloud, meshes, PC_label, queue))
    success = False
    #LM_rigid(tet_v, tet_f, point_cloud, meshes, PC_label, queue, liver_model, templates, index)
    while not success:
        try:
            min, parameter = LM_rigid(tet_v, tet_f, point_cloud, meshes, PC_label, queue, liver_model, templates, index)
            success = True
        except:
            continue
    #mlab.figure()
    '''r=animation(queue, meshes)
    process.start()
    #mlab.show()
    process.join()
    min = 1'''
    with h5py.File('../../org/datasets/Set' + str(index).zfill(3)+'/nonrigid_without_normal_separate_withoutscale_14p.h5', 'w') as f:
        f.create_dataset('error', data=min)
        f.create_dataset('parameters', data=parameter)
        f.close()
    return min, parameter

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
    liver_model = LiverModel()
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==1)[0], :])]
    #templates = 0.001*re_order_templates()
    templates = re_order_templates()
    #templates[:,4] = 0.0

    #registration
    min_all = []
    paras = []
    for i in range(1, 113):
        min, para = one_model(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces, meshes, i, liver_model, templates)
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

    #registration
    min_all = []
    paras = []
    liver_model = LiverModel()
    templates = 0.001*re_order_templates()
    templates[:,4] = 0.0
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==1)[0], :])]
    for i in range(112):
        with h5py.File('../../org/datasets/Set' + str(i+1).zfill(3) + '/non_rigid_without_normal.h5', 'r') as f:
            error = f['error'].value
            print(error)
            parameters = f['parameters'][:]
            f.close()
        with h5py.File('../../org/datasets/Set' + str(i+1).zfill(3) + '/non_rigid_with_normal.h5', 'r') as f:
            error = f['error'].value
            print(error)
            non_rigid_parameters = f['parameters'][:]
            f.close()
        min_all.append(error)
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R, C = parse_non_rigid(parameters)
        tran_v, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices, templates)

        S, T, R, C = parse_non_rigid(non_rigid_parameters)
        tran_v_non, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices, templates)
        print(i+1)
        from util_package.plot import plot_mesh_n_points_label
        plot_mesh_n_points_label(tran_v_non, meshes,\
                             point_cloud, PC_label)

    print('done')

def generate_cooresponding_points():
    min_all = []
    liver_model = LiverModel()
    templates = re_order_templates()
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces)]
    for i in range(112):
        with h5py.File('../../org/datasets/Set' + str(i+1).zfill(3) + '/rigid_without_normal_separate_withoutscale.h5', 'r') as f:
            error = f['error'].value
            print(error)
            non_rigid_parameters = f['parameters'][:]
            f.close()
        min_all.append(error)
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R, C = parse_non_rigid(non_rigid_parameters)
        tran_v_non, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices, templates)
        mesh_tri = trimesh.Trimesh(tran_v_non, liver_model.tetr_mesh.faces, process=False)
        closest_point, distance, face_id = trimesh.proximity.closest_point_naive(mesh_tri, point_cloud)
        deformation_all = np.reshape(np.matmul(templates, C), [-1, 3])
        deformation = (deformation_all[mesh_tri.faces[face_id,0], :]
                       + deformation_all[mesh_tri.faces[face_id,1], :]
                       + deformation_all[mesh_tri.faces[face_id,2], :])/3
        closest_point, rotation_matrix, deformation = apply_transform_non_rigid_reverse(S, T, R, C, point_cloud, deformation)
        print(i+1)
        #from util_package.plot import plot_mesh_n_points_label, plot_mesh_points
        #plot_mesh_n_points_label(liver_model.tetr_mesh.vertices, meshes, closest_point, PC_label)
        #plot_mesh_points(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces, closest_point)
        with h5py.File('../../org/datasets/Set' + str(i + 1).zfill(3) + '/corresponding_points_on_mesh_rigid.h5', 'w') as f:
            f.create_dataset('closest_point', data=closest_point)
            f.create_dataset('point_label', data=PC_label)
            f.create_dataset('face_id', data=face_id)

    print('done')

def re_order_templates():
    templates = load_disp_solutions('../displacement_solutions/Y1000000.0_P0.49_14p', 42)
    templates = np.reshape(templates, [-1, 3, 42])
    with h5py.File('../V_order.h5', 'r') as hf:
        order = hf['order'][:]
    #templates = templates[order, :, :]
    templates = np.reshape(templates, [-1, 42])
    return templates

def debug_44_57_67_84():
    with h5py.File('/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/src/V_order.h5', 'r') as f:
        v_order = f['order'][:]

    liver_model = LiverModel()
    mesh_liver = trimesh.load('../../org/Liver.off')

    order_reverse = np.argsort(v_order)

    deformation_templates = re_order_templates()
    deformation_templates = np.reshape(deformation_templates, [-1, 3, 14])
    for i in range(14):
        #disp = np.reshape(deformation_templates[:, i], [-1, 3])
        disp = deformation_templates[:, :, i]
        plot_meshes(liver_model.vertices, liver_model.faces,
                    liver_model.vertices+disp, liver_model.faces)

    print(' ')

if __name__ == "__main__":
    generate_cooresponding_points()