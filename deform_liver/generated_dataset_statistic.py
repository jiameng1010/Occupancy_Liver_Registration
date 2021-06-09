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
from util_package.util import load_disp_solutions
num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

def handle_punctuation(input):
    output = input.replace('.', '我')
    output = output.translate(str.maketrans('', '', string.punctuation))
    output = output.replace('我', '.')
    output = output.replace(' ', '_')
    return output

with open('generate_dataset.yaml', 'r') as f:
    cfg = yaml.load(f)

generator = DefGenerator(cfg['deformation_randomness']['Young'], cfg['deformation_randomness']['Poisson'])
deformation_all = load_disp_solutions('../displacement_solutions/Y1000000.0_P0.49', np.sum(num_of_combination[:4]))
sampler = Sampler_sparse(cfg['sample_para'])
liver_model = LiverModel()

print('generating training data')
trainingdata_cfg = 'train'+str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para'])
trainingdata_cfg = handle_punctuation(trainingdata_cfg)
train_output_dir = cfg['basics']['exp_out_dir']+trainingdata_cfg
num_of_points = []
for i in range(cfg['basics']['total_training']):
    f = h5py.File(train_output_dir+'/'+str(i).zfill(6)+'/points.h5', 'r')
    points_on = f['points_on'][:]
    points_part_label = f['points_part_label'][:]
    points_around = f['points_around'][:]
    around_inout = f['around_inout'][:]
    points_uniform = f['points_uniform'][:]
    uniform_inout = f['uniform_inout'][:]
    f.close()
    num_of_points.append(points_on.shape[0])

num_of_points = np.asarray(num_of_points)



print('generating validation data')
validationdata_cfg = 'val'+str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para'])
validationdata_cfg = handle_punctuation(validationdata_cfg)
validation_output_dir = cfg['basics']['exp_out_dir']+validationdata_cfg
for i in range(cfg['basics']['total_validation']):
    subprocess.call(['mkdir', validation_output_dir+'/'+str(i).zfill(6)])
    print('generating '+str(i)+'-th validation data')
    C = 350*np.random.normal(scale=cfg['deformation_randomness']['Gaussian_scale'], size=15)
    deformed_mesh, face_label = generator.produce_one(C)
    pymesh.save_mesh(validation_output_dir+'/'+str(i).zfill(6)+'/mesh.off', deformed_mesh)

    deformed_mesh_tri = trimesh.Trimesh(deformed_mesh.vertices, deformed_mesh.faces)
    sampler.update_deformation(deformed_mesh.vertices - liver_model.tetr_mesh.vertices)
    outputs = sampler.sample(deformed_mesh_tri, face_label)
    f = h5py.File(validation_output_dir+'/'+str(i).zfill(6)+'/points.h5', 'w')
    f.create_dataset('points_on', data=outputs[0])
    f.create_dataset('points_part_label', data=outputs[1])
    f.create_dataset('points_around', data=outputs[2])
    f.create_dataset('around_inout', data=outputs[3])
    f.create_dataset('points_uniform', data=outputs[4])
    f.create_dataset('uniform_inout', data=outputs[5])
    f.close()

print(' ')
    