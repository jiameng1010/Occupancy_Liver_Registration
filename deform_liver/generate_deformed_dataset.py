import igl
import yaml
import trimesh
import numpy as np
import copy
import os
from mayavi import mlab
from functools import reduce
from scipy import sparse
import pymesh
import h5py
import string
import subprocess
from deformation_generator import DefGenerator, Sampler_sparse
from plot import plot_points, plot_mesh, plot_mesh_points, plot_meshes_pure
from Liver import LiverModel
from util_package.util import load_disp_solutions, load_output_xyz, load_PC, angle2rotmatrix, parse_non_rigid
num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

def handle_punctuation(input):
    output = input.replace('.', '我')
    output = output.translate(str.maketrans('', '', string.punctuation))
    output = output.replace('我', '.')
    output = output.replace(' ', '_')
    return output

def apply_transform_rigid_reverse(S, T, R, tet_v):
    r1, r2, r3, rr = angle2rotmatrix(R)
    invert_rr = np.linalg.inv(rr)
    tran_v = tet_v - np.expand_dims(T, axis=0)
    tran_v = tran_v / S
    tran_v = np.matmul(invert_rr, tran_v.T).T
    return tran_v, invert_rr

with open('generate_dataset.yaml', 'r') as f:
    cfg = yaml.load(f)

generator = DefGenerator(cfg['deformation_randomness']['Young'], cfg['deformation_randomness']['Poisson'],
                         '../displacement_solutions/Y1000000.0_P0.49_demean', 14)
sampler = Sampler_sparse(cfg['sample_para'])
liver_model = LiverModel()

print('generating training data')
trainingdata_cfg = 'train'+str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para'])
trainingdata_cfg = handle_punctuation(trainingdata_cfg)
train_output_dir = cfg['basics']['exp_out_dir']+trainingdata_cfg
if not os.path.isdir(train_output_dir):
    subprocess.call(['mkdir', train_output_dir])
    for i in range(cfg['basics']['total_training']):
        subprocess.call(['mkdir', train_output_dir+'/'+str(i).zfill(6)])
        print('generating '+str(i)+'-th training data')
        C = 300*np.random.normal(scale=cfg['deformation_randomness']['Gaussian_scale'], size=14)
        deformed_mesh, face_label = generator.produce_one(C)
        pymesh.save_mesh(train_output_dir+'/'+str(i).zfill(6)+'/mesh.off', deformed_mesh)

        '''h5f = h5py.File(
            '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/src/deform_liver/tetr_mesh.h5', 'r')
        vertices = h5f['mesh_vertices'][:]
        faces = h5f['mesh_faces'][:]
        plot_meshes_pure(vertices, faces, deformed_mesh.vertices, deformed_mesh.faces)'''

        deformed_mesh_tri = trimesh.Trimesh(deformed_mesh.vertices, deformed_mesh.faces)
        sampler.update_deformation(deformed_mesh.vertices - liver_model.tetr_mesh.vertices)
        outputs = sampler.sample(deformed_mesh_tri, face_label)
        f = h5py.File(train_output_dir+'/'+str(i).zfill(6)+'/points.h5', 'w')
        f.create_dataset('points_on', data=outputs[0])
        f.create_dataset('points_part_label', data=outputs[1])
        f.create_dataset('points_around', data=outputs[2])
        f.create_dataset('around_inout', data=outputs[3])
        f.create_dataset('points_uniform', data=outputs[4])
        f.create_dataset('uniform_inout', data=outputs[5])
        f.close()

        #from util_package.plot import plot_meshes_n_points
        #plot_meshes_n_points(deformed_mesh.vertices, deformed_mesh.faces,
        #                     liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces,
        #                     liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces, outputs[0])
        #plot_mesh_points(deformed_mesh.vertices, deformed_mesh.faces, outputs[2])

print('generating validation data')
validationdata_cfg = 'val'+str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para'])
validationdata_cfg = handle_punctuation(validationdata_cfg)
validation_output_dir = cfg['basics']['exp_out_dir']+validationdata_cfg
if not os.path.isdir(validation_output_dir):
    subprocess.call(['mkdir', validation_output_dir])
    for i in range(112):
        subprocess.call(['mkdir', validation_output_dir+'/'+str(i).zfill(6)])
        print('generating '+str(i)+'-th validation data')
        #C = 350*np.random.normal(scale=cfg['deformation_randomness']['Gaussian_scale'], size=15)
        #deformed_mesh, face_label = generator.produce_one(C)

        displacement = load_output_xyz('../../org/reg_out/ResultsSet' + str(i+1).zfill(3) + '.xyz')
        reg_vert = liver_model.vertices + displacement
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        mean = np.expand_dims(np.mean(point_cloud, axis=0), axis=0) / 1000
        reg_vert = reg_vert - mean
        with h5py.File('../../org/datasets/Set' + str(i+1).zfill(3) + '/nonrigid_without_normal_separate_withoutscale_14p.h5', 'r') as f:
            non_rigid_parameters = f['parameters'][:]
        S, T, R, C = parse_non_rigid(non_rigid_parameters)
        tran_v, _ = apply_transform_rigid_reverse(S, T, R, reg_vert)
        deformed_mesh = pymesh.form_mesh(tran_v, liver_model.faces)
        pymesh.save_mesh(validation_output_dir+'/'+str(i).zfill(6)+'/mesh.off', deformed_mesh)

        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        mean = np.expand_dims(np.mean(point_cloud, axis=0), axis=0) / 1000

        deformed_mesh_tri = trimesh.Trimesh(deformed_mesh.vertices, deformed_mesh.faces)
        sampler.update_deformation(deformed_mesh.vertices - liver_model.tetr_mesh.vertices)
        outputs = sampler.sample(deformed_mesh_tri, liver_model.face_label, val_index=i)
        f = h5py.File(validation_output_dir+'/'+str(i).zfill(6)+'/points.h5', 'w')
        points_on, _ = apply_transform_rigid_reverse(S, T, R, outputs[0]-mean)
        f.create_dataset('points_on', data=points_on)
        f.create_dataset('points_part_label', data=outputs[1])
        f.create_dataset('points_around', data=outputs[2])
        f.create_dataset('around_inout', data=outputs[3])
        f.create_dataset('points_uniform', data=outputs[4])
        f.create_dataset('uniform_inout', data=outputs[5])
        f.close()

        '''point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i+1).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        from util_package.plot import plot_meshes_n_points, plot_pointss
        plot_pointss(point_cloud, outputs[0])
        plot_meshes_n_points(deformed_mesh.vertices, deformed_mesh.faces,
                             liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces,
                             liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces, outputs[0])
        plot_mesh_points(deformed_mesh.vertices, deformed_mesh.faces, outputs[2])'''

print(' ')
    