import pymesh
import h5py
import numpy as np
import scipy
import cupy as cp
import sys
import trimesh
import os
import copy
#from mayavi import mlab
from pymesh import Material
from trimesh import sample
#from debug_plot import plot_displacement, plot_mesh_surfacenormal
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from util_package.util import load_voxel_mesh, mesh_volume, load_disp_solutions, load_PC
from plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, plot_meshes, \
    plot_3meshes
from LM_fit.solver import solve_one_laplacian, solve_one,\
    solve_one_stiffness, solve_one_elastic, solve_one_quadratic,\
    solve_one_quadratic_new
from Liver import LiverModel

num_of_combination = np.asarray([1,2,3,4,5,6,7,8])
order_of_polynomial = int(4)

def stiffness_matrix():
    mesh = pymesh.load_mesh('../../org/Liver.off')
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    tetr_mesh = pymesh.tetrahedralize(mesh, cell_size=0.008, engine='cgal_no_features')
    print(tetr_mesh.voxels.shape)
    #tetr_mesh = load_voxel_mesh()

    young = 1000.0
    poisson = 0.3
    mat = Material.create_isotropic(3, 1.0, young, poisson)
    assembler = pymesh.Assembler(tetr_mesh, material=mat)
    Stiff = assembler.assemble("stiffness")
    plot_mesh_points(tetr_mesh.vertices, tetr_mesh.faces, tetr_mesh.vertices)
    return tetr_mesh, Stiff

def generate_complement(mesh_all, mesh):
    v = mesh_all.vertices
    f_all = mesh_all.faces
    f = mesh.faces
    #mesh_tri = trimesh.Trimesh(vertices=v, faces=f_all)
    #mesh_tri.show()

    a0 = np.isin(f_all[:, 0], f[:, 0])
    a1 = np.isin(f_all[:, 1], f[:, 1])
    a2 = np.isin(f_all[:, 2], f[:, 2])
    a = np.logical_not(np.logical_and(a0, np.logical_and(a1, a2)))
    f_c = f_all[a, :]
    mesh_c = pymesh.form_mesh(v, f_c)
    mesh_c, info = pymesh.remove_isolated_vertices(mesh_c)
    return mesh_c, info["ori_vertex_index"]

def linear_PCA(points):
    mean = np.mean(points, axis=0, keepdims=True)
    de_meaned = points - mean
    covariance_matrix = np.matmul(de_meaned.T, de_meaned)
    pca_vectors = np.linalg.eig(covariance_matrix)
    return mean, pca_vectors

def get_displacement_boundary(C_matrix, vertices, pca):
    x = np.matmul(vertices, pca[:,0])
    y = np.matmul(vertices, pca[:,1])
    d = np.polynomial.polynomial.polyval2d(x,y, 10*C_matrix)
    displacement = np.tile(np.expand_dims(d, axis=1), [1,3]) * pca[:,2]
    displacement = displacement - np.mean(displacement, axis=0, keepdims=True)
    #plot_displacement(mesh_back, displacement)
    return displacement, np.max(d)-np.min(d)

def load_stiff(young, poisson):
    h5f = h5py.File('tetr_mesh.h5', 'r')
    vertices = h5f['mesh_vertices'][:]
    faces = h5f['mesh_faces'][:]
    voxels = h5f['mesh_voxels'][:]
    #inv_stiff = h5f['inv_stiffness'][:]
    #stiff = h5f['stiffness'][:]
    h5f.close()
    mesh = pymesh.form_mesh(vertices, faces, voxels)
    mat = Material.create_isotropic(3, 1.0, young, poisson)
    assembler = pymesh.Assembler(mesh, material=mat)
    Stiff = assembler.assemble("stiffness")
    Laplacian = assembler.assemble("laplacian")
    laplacian = Laplacian
    return mesh, Stiff, laplacian

def get_back_area(mesh, mesh_back, mesh_front):
    mesh_back_tri = trimesh.Trimesh(mesh_back.vertices, mesh_back.faces)
    mesh_front_i, _ = pymesh.remove_isolated_vertices(mesh_front)
    mesh_front_tri = trimesh.Trimesh(mesh_front_i.vertices, mesh_front_i.faces)
    surface, index = pymesh.remove_isolated_vertices(pymesh.form_mesh(mesh.vertices, mesh.faces))
    d_back = np.abs(trimesh.proximity.signed_distance(mesh_back_tri, surface.vertices))
    d_front = np.abs(trimesh.proximity.signed_distance(mesh_front_tri, surface.vertices))
    back = 1*np.where(d_front > d_back)
    back = index['ori_vertex_index'][back[0]]
    return back

def generate_displacement(mesh, C, pca_vectors):
    C_matrix = np.zeros(shape=(order_of_polynomial+1, order_of_polynomial+1))
    iu1 = np.triu_indices(5)
    C_matrix[iu1] = C
    C_matrix = np.fliplr(C_matrix)
    vertices_sorted = copy.copy(mesh.vertices)
    displacement, range = get_displacement_boundary(C_matrix, vertices_sorted, pca_vectors[1])
    return displacement, range

def compute_displacements(young, poisson, saving_dir):
    elastic_cfg = 'Y' + str(young) + '_' + 'P' + str(poisson)
    os.popen('mkdir ../displacement_solutions/'+elastic_cfg)
    mesh_back, boundary_vindice = generate_complement(pymesh.load_mesh('../../org/Liver.off'), pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh(mesh_back.vertices, mesh_back.faces)
    mean, pca_vectors = linear_PCA(mesh_back.vertices)
    #plot_mesh_surfacenormal(mesh_back.vertices, mesh_back.faces, mean, pca_vectors[1])

    mesh, stiff, laplacian = load_stiff(young, poisson)
    back_index = get_back_area(mesh, mesh_back, pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh_points(mesh.vertices, mesh.faces, mesh.vertices[back_index, :])

    C = np.random.normal(scale=0.02, size=np.sum(num_of_combination[:(order_of_polynomial+1)]))
    displacement, ranges = generate_displacement(mesh, C, pca_vectors)
    #displacement, ranges = generate_displacement(mesh, C, pca_vectors)
    with h5py.File('coefficient.h5', 'r') as hf:
        coefficient = hf['coefficient'][:]
        coefficient[4] = 0.0
    for i in range(np.sum(num_of_combination[:(order_of_polynomial+1)])):
        C = np.zeros(shape=[np.sum(num_of_combination[:(order_of_polynomial+1)])])
        C[i] = 1.0
        C = 0.05 * C * coefficient
        C[4] = 0.0
        displacement, ranges = generate_displacement(mesh, C, pca_vectors)
        mesh_deformed = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
        displacement = np.reshape(mesh_deformed.vertices - mesh.vertices,
                                  [-1])
        with h5py.File('../displacement_solutions/'+elastic_cfg+'/' + str(i).zfill(3) + '.h5', 'w') as hf:
            hf.create_dataset("displacement", data=displacement)
            
            
    '''print("volume_org: " + str(mesh_volume(mesh)))
    mesh_deformed1 = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
    print("volume1: " + str(mesh_volume(mesh_deformed1)))
    mesh_deformed2 = solve_one_stiffness(mesh, stiff, back_index, displacement)
    print("volume2: " + str(mesh_volume(mesh_deformed2)))
    mesh_deformed3 = solve_one_elastic(mesh, stiff, laplacian, back_index, displacement)
    print("volume3: " + str(mesh_volume(mesh_deformed3)))
    plot_3meshes(mesh_deformed1.vertices, mesh_deformed1.faces,
                 mesh_deformed2.vertices, mesh_deformed2.faces,
                 mesh_deformed3.vertices, mesh_deformed3.faces)
    plot_meshes(mesh_deformed2.vertices, mesh_deformed2.faces,
                mesh_deformed3.vertices, mesh_deformed3.faces, back_index)'''


    '''young = [1.0e4]
    poisson = [-0.95, -0.9, -0.7, -0.5, -0.3, -0.1, -0.0, -0.2, -0.4, 0.45]
    deformed_meshes = []
    for i in range(len(poisson)):
        mesh, stiff, laplacian = load_stiff(young[0], poisson[i])
        print('young: ' + str(young[0]) + "   "\
              'poisson: ' + str(poisson[i]))
        mesh_deformed = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
        deformed_meshes.append(np.reshape((mesh_deformed.vertices-mesh.vertices)))
    print(' ')'''

def main():
    mesh_back, boundary_vindice = generate_complement(pymesh.load_mesh('../../org/Liver.off'), pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh(mesh_back.vertices, mesh_back.faces)
    mean, pca_vectors = linear_PCA(mesh_back.vertices)
    #plot_mesh_surfacenormal(mesh_back.vertices, mesh_back.faces, mean, pca_vectors[1])

    mesh, stiff, laplacian = load_stiff(1.0e6, 0.49)
    back_index = get_back_area(mesh, mesh_back, pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh_points(mesh.vertices, mesh.faces, mesh.vertices[back_index, :])
    mesh_liver = trimesh.load('../../org/Liver.off')

    #displacement, ranges = generate_displacement(mesh, C, pca_vectors)
    with h5py.File('coefficient.h5', 'r') as hf:
        coefficient = hf['coefficient'][:]
    displacements = load_disp_solutions('../displacement_solutions/Y1.0e4_P0.4', np.sum(num_of_combination[:(order_of_polynomial+1)]))
    CC = np.random.normal(scale=0.3, size=np.sum(num_of_combination[:(order_of_polynomial + 1)]))
    C = 0.05 * CC * coefficient
    C[4] = 0.0
    displacement, ranges = generate_displacement(mesh, C, pca_vectors)

    approxi_displacement = np.reshape(np.matmul(displacements, CC), [-1, 3])
    mesh_deformed2 = pymesh.form_mesh(mesh.vertices+approxi_displacement, mesh.faces)
    plot_meshes(mesh.vertices, mesh.faces,
                mesh_deformed2.vertices, mesh_deformed2.faces, back_index)
    plot_mesh_points(mesh_deformed2.vertices, mesh_deformed2.faces, mesh.vertices+displacement)

    mesh_deformed1 = solve_one_quadratic_new(mesh, stiff, laplacian, back_index, displacement)
    plot_meshes(mesh_deformed1.vertices, mesh_deformed1.faces,
                mesh_deformed2.vertices, mesh_deformed2.faces, back_index)

class DefGenerator():
    def __init__(self, young, poisson, elastic_cfg, num_solution):
        if os.path.isdir('../displacement_solutions/'+elastic_cfg):
            print('linear elastic solution exist, load form disk ...')
            self.displacements = load_disp_solutions('../displacement_solutions/'+elastic_cfg,
                                            num_solution)
            self.displacements[:,4] = 0.0
            self.displacements = 0.001 * self.displacements
        else:
            print('linear elastic solution do not exist, generating with quadratic solver ...')
            compute_displacements(young, poisson, '../displacement_solutions/'+elastic_cfg)

        h5f = h5py.File('tetr_mesh.h5', 'r')
        vertices = h5f['mesh_vertices'][:]
        faces = h5f['mesh_faces'][:]
        voxels = h5f['mesh_voxels'][:]
        # inv_stiff = h5f['inv_stiffness'][:]
        # stiff = h5f['stiffness'][:]
        h5f.close()
        self.mesh = pymesh.form_mesh(vertices, faces, voxels)
        with h5py.File('new_face_label.h5', 'r') as f:
            self.f_label = f['face_label'][:]

    def produce_one(self, random_coefficients):
        approxi_displacement = np.reshape(np.matmul(self.displacements, random_coefficients), [-1, 3])
        mesh_deformed = pymesh.form_mesh(self.mesh.vertices + approxi_displacement, self.mesh.faces)
        #back_index = get_back_area(self.mesh, mesh_back, pymesh.load_mesh('../../org/Liver_Front.off'))
        return mesh_deformed, self.f_label

class Sampler():
    def __init__(self, cfg):
        self.scale = cfg['Gaussian_scale']
        self.num_onsurface_points = cfg['num_onsurface_points']
        self.num_around_surface = cfg['num_around_surface']
        self.num_uniform = cfg['num_uniform']
        self.bbox_padding = cfg['bbox_padding']

    def sample(self, mesh, label):
        points_on, faces_id = sample.sample_surface(mesh, self.num_onsurface_points)
        points_part_label = label[faces_id]

        points_around, faces_id = sample.sample_surface(mesh, self.num_around_surface)
        points_around = points_around + np.random.normal(scale=self.scale, size=points_around.shape)
        around_inout = mesh.contains(points_around)

        bbox = mesh.vertices[:,0]
        range = max(bbox) - min(bbox)
        points_uniform0 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,1]
        range = max(bbox) - min(bbox)
        points_uniform1 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,2]
        range = max(bbox) - min(bbox)
        points_uniform2 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        points_uniform = np.concatenate([np.expand_dims(points_uniform0, axis=1),
                                         np.expand_dims(points_uniform1, axis=1),
                                         np.expand_dims(points_uniform2, axis=1)],
                                         axis=1)
        uniform_inout = mesh.contains(points_uniform)
        return points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout

def closest_point_onsurface(id, structure_label):
    points_filename = '../../org/datasets/Set' + str(id).zfill(3) + '/corresponding_points_on_mesh_rigid.h5'
    with h5py.File(points_filename, 'r') as f:
        closest_point = f['closest_point'][:]
        point_label = f['point_label'][:]
        face_id = f['face_id'][:]
    return closest_point[np.where(point_label==structure_label)[0], :], face_id[np.where(point_label==structure_label)[0]]

def closest_point_onsurface_val(id, structure_label):
    points_filename = '../../org/datasets/Set' + str(id).zfill(3) + '/corresponding_points_on_mesh_rigid.h5'
    point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(id).zfill(3))
    #point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = point_cloud / 1000
    with h5py.File(points_filename, 'r') as f:
        closest_point = f['closest_point'][:]
        point_label = f['point_label'][:]
        face_id = f['face_id'][:]
    return point_cloud[np.where(point_label==structure_label)[0], :], face_id[np.where(point_label==structure_label)[0]]

def get_sparse_points_fromone(liver_model, deformation_all, val_index):
    point_FF, face_idFF = closest_point_onsurface_val(val_index, 2)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idFF, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idFF, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idFF, 2], :]) / 3
    #point_FF = point_FF + deformation
    point_LR, face_idLR = closest_point_onsurface_val(val_index, 3)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idLR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idLR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idLR, 2], :]) / 3
    #point_LR = point_LR + deformation
    point_RR, face_idRR = closest_point_onsurface_val(val_index, 4)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idRR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idRR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idRR, 2], :]) / 3
    #point_RR = point_RR + deformation
    point_SR, face_idSR = closest_point_onsurface_val(val_index, 1)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idSR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idSR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idSR, 2], :]) / 3
    #point_SR = point_SR + deformation

    points_on = np.concatenate([point_SR, point_FF, point_LR, point_RR], axis=0)
    points_part_label = np.concatenate([1*np.ones_like(point_SR), 2*np.ones_like(point_FF), 3*np.ones_like(point_LR), 4*np.ones_like(point_RR),], axis=0)
    return points_on, points_part_label[:, 0]

def get_sparse_points(liver_model, deformation_all):
    point_FF, face_idFF = closest_point_onsurface(round(112 * np.random.uniform(0.0, 1.0) + 0.5), 2)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idFF, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idFF, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idFF, 2], :]) / 3
    point_FF = point_FF + deformation
    point_LR, face_idLR = closest_point_onsurface(round(112 * np.random.uniform(0.0, 1.0) + 0.5), 3)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idLR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idLR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idLR, 2], :]) / 3
    point_LR = point_LR + deformation
    point_RR, face_idRR = closest_point_onsurface(round(112 * np.random.uniform(0.0, 1.0) + 0.5), 4)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idRR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idRR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idRR, 2], :]) / 3
    point_RR = point_RR + deformation
    point_SR, face_idSR = closest_point_onsurface(round(112 * np.random.uniform(0.0, 1.0) + 0.5), 1)
    deformation = (deformation_all[liver_model.tetr_mesh.faces[face_idSR, 0], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idSR, 1], :]
                   + deformation_all[liver_model.tetr_mesh.faces[face_idSR, 2], :]) / 3
    point_SR = point_SR + deformation

    points_on = np.concatenate([point_SR, point_FF, point_LR, point_RR], axis=0)
    points_part_label = np.concatenate([1*np.ones_like(point_SR), 2*np.ones_like(point_FF), 3*np.ones_like(point_LR), 4*np.ones_like(point_RR),], axis=0)
    return points_on, points_part_label[:, 0]

class Sampler_sparse():
    def __init__(self, cfg):
        self.scale = cfg['Gaussian_scale']
        self.num_onsurface_points = cfg['num_onsurface_points']
        self.num_around_surface = cfg['num_around_surface']
        self.num_uniform = cfg['num_uniform']
        self.bbox_padding = cfg['bbox_padding']
        self.liver_model = LiverModel()

    def update_deformation(self, deformation):
        self.deformation = deformation

    def sample(self, mesh, label, val_index=-1):
        if val_index != -1:
            points_on, points_part_label = get_sparse_points_fromone(self.liver_model, self.deformation, val_index+1)
        else:
            points_on, points_part_label = get_sparse_points(self.liver_model, self.deformation)
            #points_on = points_on + np.random.normal(0.0, 1e-3, size=points_on.shape)

        points_around, faces_id = sample.sample_surface(mesh, self.num_around_surface)
        points_around = points_around + np.random.normal(scale=self.scale, size=points_around.shape)
        around_inout = mesh.contains(points_around)

        bbox = mesh.vertices[:,0]
        range = max(bbox) - min(bbox)
        points_uniform0 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,1]
        range = max(bbox) - min(bbox)
        points_uniform1 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,2]
        range = max(bbox) - min(bbox)
        points_uniform2 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        points_uniform = np.concatenate([np.expand_dims(points_uniform0, axis=1),
                                         np.expand_dims(points_uniform1, axis=1),
                                         np.expand_dims(points_uniform2, axis=1)],
                                         axis=1)
        uniform_inout = mesh.contains(points_uniform)
        return points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout

def faces_label():
    h5f = h5py.File('tetr_mesh.h5', 'r')
    vertices = h5f['mesh_vertices'][:]
    faces = h5f['mesh_faces'][:]
    voxels = h5f['mesh_voxels'][:]
    h5f.close()
    mesh = pymesh.form_mesh(vertices, faces, voxels)
    label = np.zeros(shape=mesh.faces.shape[0], dtype=np.int16)

    #distance = np.expand_dims(mesh.vertices, axis=1) - np.expand_dims(mesh_Front.vertices, axis=0)
    #distance = np.sum(distance * distance, axis=2)
    #map = np.argmin(distance, axis=1)

    f = h5py.File('vertices_map.hdf5', 'r')
    map = f['vertices_map'][:]
    f.close()

    mesh_FF = trimesh.load('../../org/Liver_FF.off')
    mesh_LR = trimesh.load('../../org/Liver_LR.off')
    mesh_RR = trimesh.load('../../org/Liver_RR.off')
    mesh_Front = trimesh.load('../../org/Liver_Front.off')
    face_indices = find_faces(mesh, map, mesh_Front)
    label[face_indices] = 1
    face_indices = find_faces(mesh, map, mesh_FF)
    label[face_indices] = 2
    face_indices = find_faces(mesh, map, mesh_LR)
    label[face_indices] = 3
    face_indices = find_faces(mesh, map, mesh_RR)
    label[face_indices] = 4

    f = h5py.File('face_label.hdf5', 'w')
    f.create_dataset('face_label', data=label)
    f.close()

    print('done')

def find_faces(mesh, map, mesh_t):
    f_all = map[mesh.faces]
    f = mesh_t.faces
    # mesh_tri = trimesh.Trimesh(vertices=v, faces=f_all)
    # mesh_tri.show()

    a0 = np.isin(f_all[:, 0], f[:, 0])
    a1 = np.isin(f_all[:, 1], f[:, 1])
    a2 = np.isin(f_all[:, 2], f[:, 2])
    a = np.logical_and(a0, np.logical_and(a1, a2))
    return np.where(a==True)


if __name__ == "__main__":
    #compute_displacements(1.0e4, 0.4)
    #g = DefGenerator(1.0e6, 0.49)
    main()






