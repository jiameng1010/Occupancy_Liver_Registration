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
    parse_non_rigid, normalize, load_voxel_mesh, load_xyz, closest
from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label,\
    put_queue, plot_mesh_points_label_anim, update_scene_objs_anim, plot_meshes_n_points, plot_meshes

from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point
num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

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

def plot_reg():
    #registration
    min_all = []
    paras = []
    liver_model = LiverModel()
    #liver_org = load_voxel_mesh()
    templates = re_order_templates()
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces[np.where(liver_model.face_label==1)[0], :])]
    for i in range(1, 113):#[33, 67, 57, 44, 84, 11, 22, 78, 91, 100, 110, 5]:
        with h5py.File('../../org/datasets/Set' + str(i).zfill(3) + '/rigid_without_normal_separate_withoutscale.h5', 'r') as f:
            error = f['error'].value
            print(error)
            non_rigid_parameters = f['parameters'][:]
            f.close()
        min_all.append(error)
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        mean = np.expand_dims(np.mean(point_cloud, axis=0), axis=0) / 1000
        #point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R, C = parse_non_rigid(non_rigid_parameters)
        tran_v_non, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices, templates)
        tran_v_non = tran_v_non + mean

        #S, T, R = parse(non_rigid_parameters)
        #tran_v_non, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, np.zeros(shape=14), liver_model.tetr_mesh.vertices, templates)
        #tran_v_non = tran_v_non + mean

        displacement = tran_v_non - liver_model.tetr_mesh.vertices
        save_xyz_file('../../org/datasets/Set' + str(i).zfill(3) + '/ResultsSet' + str(i).zfill(3) + '.xyz', displacement)
        save_xyz_file('../../org/reg_out/ResultsSet' + str(i).zfill(3) + '.xyz', displacement)
        print(i+1)
        #from util_package.plot import plot_mesh_n_points_label
        #plot_mesh_n_points_label(tran_v_non, meshes,\
        #                     point_cloud, PC_label)

    print('done')


def save_xyz_file(filename, displacement):
    f_org = open('../../org/LiverVolume.nod', 'r')
    f_out = open(filename, 'w')
    for i in range(29545):
        half = f_org.readline()
        half2 = '{:f}'.format(displacement[i, 0]) + ' ' + '{:f}'.format(displacement[i, 1]) + ' ' + '{:f}'.format(displacement[i, 2])
        f_out.write(half[:-1] + ' ' + half2 + '\n')
    f_org.close()
    f_out.close()



def re_order_templates():
    templates = load_disp_solutions('../displacement_solutions/Y1000000.0_P0.49_14p', 42)
    templates = np.reshape(templates, [-1, 3, 42])
    with h5py.File('../V_order.h5', 'r') as hf:
        order = hf['order'][:]
    #templates = templates[order, :, :]
    templates = np.reshape(templates, [-1, 42])
    return templates

def eval_44_57_67_84():
    from util_package.util import tetra_interpolation_full
    liver_model = LiverModel()
    templates = re_order_templates()
    target_dir = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/AdditionalDataForParticipants'
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 1)[0], :])]
    for i in [44, 67, 57, 84]:
        with h5py.File(
                '../../org/datasets/Set' + str(i).zfill(3) + '/nonrigid_without_normal_separate_withoutscale_14p.h5',
                'r') as f:
            error = f['error'].value
            non_rigid_parameters = f['parameters'][:]
            f.close()
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        mean = np.expand_dims(np.mean(point_cloud, axis=0), axis=0) / 1000
        # point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R, C = parse_non_rigid(non_rigid_parameters)
        tran_v_non, rotation_matrix, deformation = apply_transform_non_rigid(S, T, R, C, liver_model.tetr_mesh.vertices,
                                                                             templates)
        tran_v_non = tran_v_non + mean

        targets_org = (1 / 1000) * load_xyz(target_dir + '/LiverTargetsIncomplete.xyz')
        _, closet_vert_target = closest(targets_org, liver_model.vertices)

        GT_target = load_xyz(target_dir + '/Set' + str(i).zfill(3) + '_TargetsIncomplete.xyz')
        GT_target = (1 / 1000) * GT_target

        output_target = tetra_interpolation_full(targets_org, liver_model.vertices, tran_v_non)
        difference = GT_target - output_target
        difference_m = np.sqrt(np.sum(difference*difference, axis=1))
        mean_error = np.mean(np.sqrt(np.sum(difference * difference, axis=1)))
        import matplotlib.pyplot as plt
        plt.boxplot([difference_m,difference_m],
                    label= ['baseline-rigid', 'baseline-nonrigid'])
        plt.show()
        print(i)
        print(mean_error)
        from util_package.plot import plot_mesh_n_points_label
        plot_mesh_n_points_label(tran_v_non, meshes,\
                             point_cloud, PC_label)

    print('done')

#plot_reg()

eval_44_57_67_84()