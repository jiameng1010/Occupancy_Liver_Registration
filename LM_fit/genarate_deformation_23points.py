import pymesh
import h5py
import numpy as np
import scipy
import cupy as cp
import trimesh
import copy
from pymesh import Material
from plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, plot_meshes, \
    plot_3meshes
from debug_plot import plot_displacement, plot_mesh_surfacenormal
from util import load_voxel_mesh, mesh_volume, load_disp_solutions
from Liver import LiverModel
from solver import solve_one_laplacian, solve_one,\
    solve_one_stiffness, solve_one_elastic, solve_one_quadratic,\
    solve_one_elastic_new, solve_one_quadratic_new

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

def get_displacement_boundary(C_matrix, vertices, mean, pca, index_in_org):
    x = 500*np.matmul(vertices-mean, pca[:,0])
    x = x - np.min(x)
    y = 750*np.matmul(vertices-mean, pca[:,1])
    y = y - np.min(y)
    d = np.polynomial.polynomial.polyval2d(x,y, C_matrix)
    d = (d - np.mean(d[index_in_org]))/(np.max(d)-np.min(d))
    displacement = np.tile(np.expand_dims(d, axis=1), [1,3]) * pca[:,2]
    #displacement = displacement - np.mean(displacement, axis=0, keepdims=True)
    #plot_displacement(mesh_back, displacement)
    return displacement/40, np.max(d)-np.min(d)

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

def find_control_points(liver_model):
    mesh_complement = pymesh.load_mesh('../../org/Liver_surface_23points.off')
    distance = np.expand_dims(liver_model.surface_mesh.vertices, axis=0) - np.expand_dims(mesh_complement.vertices, axis=1)
    distance = np.sum(distance * distance, axis=2)
    min_distance = np.min(distance, axis=0)
    return np.where(min_distance != 0)[0]

def generate_displacement(liver_model, mesh, C, mean, pca_vectors):
    control_point_index = find_control_points(liver_model)
    control_point_index_in_org = liver_model.surface_nodes_orgindex[control_point_index]
    displacement = np.expand_dims(pca_vectors[1][:,0], axis=0) * np.expand_dims(C, axis=1) #######################[1][:, (0, 1, 2(most))]
    return displacement/1000, control_point_index_in_org

'''def compute_displacements():
    mesh_back, boundary_vindice = generate_complement(pymesh.load_mesh('../../org/Liver.off'), pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh(mesh_back.vertices, mesh_back.faces)
    mean, pca_vectors = linear_PCA(mesh_back.vertices)
    #plot_mesh_surfacenormal(mesh_back.vertices, mesh_back.faces, mean, pca_vectors[1])

    mesh, stiff, laplacian = load_stiff(1.0e4, 0.4)
    back_index = get_back_area(mesh, mesh_back, pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh_points(mesh.vertices, mesh.faces, mesh.vertices[back_index, :])

    with h5py.File('coefficient.h5', 'r') as hf:
        coefficient = hf['coefficient'][:]
    for i in range(23):
        C = np.zeros(shape=[23])
        C[i] = 1.0
        C = 0.05 * C * coefficient
        C[4] = 0.0
        displacement, ranges = generate_displacement(mesh, C, pca_vectors)
        mesh_deformed = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
        displacement = np.reshape(mesh_deformed.vertices - mesh.vertices,
                                  [-1])
        with h5py.File('../displacement_solutions/Y1e4_P04/' + str(i).zfill(3) + '.h5', 'w') as hf:
            hf.create_dataset("displacement", data=displacement)
            
            
    print("volume_org: " + str(mesh_volume(mesh)))
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
                mesh_deformed3.vertices, mesh_deformed3.faces, back_index)


    young = [1.0e4]
    poisson = [-0.95, -0.9, -0.7, -0.5, -0.3, -0.1, -0.0, -0.2, -0.4, 0.45]
    deformed_meshes = []
    for i in range(len(poisson)):
        mesh, stiff, laplacian = load_stiff(young[0], poisson[i])
        print('young: ' + str(young[0]) + "   "\
              'poisson: ' + str(poisson[i]))
        mesh_deformed = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
        deformed_meshes.append(np.reshape((mesh_deformed.vertices-mesh.vertices)))
    print(' ')'''

def produce_back_index(back_index, liver_model):
    back_indicator = np.zeros(shape=liver_model.tetr_mesh.vertices.shape[0], dtype=np.int16)
    back_indicator[back_index] = 1

    distance = np.expand_dims(liver_model.tetr_mesh.vertices, axis=1) - np.expand_dims(liver_model.simplified_tetr_mesh.vertices, axis=0)
    distance = np.sum(distance*distance, axis=2)
    index_in_org = np.argmin(distance, axis=0)
    simplified_back_indicator = back_indicator[index_in_org]
    return np.where(simplified_back_indicator==1)[0], index_in_org


def main():
    mesh_back, boundary_vindice = generate_complement(pymesh.load_mesh('../../org/Liver.off'), pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh(mesh_back.vertices, mesh_back.faces)
    mean, pca_vectors = linear_PCA(mesh_back.vertices)
    #plot_mesh_surfacenormal(mesh_back.vertices, mesh_back.faces, mean, pca_vectors[1])

    mesh, stiff, laplacian = load_stiff(1.0e6, 0.49)

    liver_model = LiverModel()
    from util_package.plot import plot_mesh_vectors, plot_mesh_points_vector
    #plot_mesh_vectors(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
    #                        mean, pca_vectors[1][:,2])
    liver_model.build_stiffness()

    for i in range(23):
        C = np.zeros(shape=[23])
        C[i] = 1.0
        displacement, control_point_index = generate_displacement(liver_model, mesh, C, mean, pca_vectors)
        back_index_simplified, _ = produce_back_index(control_point_index, liver_model)
        #plot_mesh_vectors(liver_model.tetr_mesh.vertices, liver_model.tetr_mesh.faces,
        #                  liver_model.tetr_mesh.vertices[control_point_index, :],
        #                  displacement[:, :])
        print('volume: ' + str(mesh.volume))
        all_displacement = np.zeros_like(liver_model.tetr_mesh.vertices)
        all_displacement[control_point_index, :] = displacement
        sim_displacement = np.zeros_like(liver_model.simplified_tetr_mesh.vertices)
        sim_displacement[back_index_simplified, :] = displacement


        #plot_mesh_vectors(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
        #                        liver_model.simplified_tetr_mesh.vertices[back_index_simplified, :], displacement[:, :])
        #continue
        simplified_mesh_deformed2 = solve_one_quadratic(liver_model.tetr_mesh,
                                                           liver_model.stiffness,
                                                           laplacian, control_point_index,
                                                           all_displacement)
        simplified_mesh_deformed = solve_one_quadratic_new(liver_model.simplified_tetr_mesh,
                                                           liver_model.simplified_stiffness,
                                                           laplacian, back_index_simplified,
                                                           sim_displacement)
        #plot_meshes(simplified_mesh_deformed2.vertices, simplified_mesh_deformed2.faces,
        #            simplified_mesh_deformed.vertices, simplified_mesh_deformed.faces, back_index_simplified)
        #return

        #mesh_deformed1 = solve_one_quadratic_new(mesh, stiff, laplacian, back_index, displacement)
        #pymesh.save_mesh('gnearated_mesh.off', mesh_deformed1)
        #mesh_deformed1 = solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement)
        simplified_displacement = simplified_mesh_deformed.vertices - liver_model.simplified_tetr_mesh.vertices
        full_displacement = simplified_mesh_deformed2.vertices - liver_model.tetr_mesh.vertices
        '''from util_package.util import tetra_interpolation
        full_displacement, points_not_inside_trihull = tetra_interpolation(mesh.vertices,
                                                liver_model.simplified_tetr_mesh.vertices,
                                                simplified_displacement,
                                                triangulation)
        distance = np.expand_dims(liver_model.simplified_tetr_mesh.vertices, axis=1) - np.expand_dims(
            mesh.vertices[points_not_inside_trihull, :], axis=0)
        distance = np.sum(distance * distance, axis=2)
        bad_index_in_org = np.argmin(distance, axis=0)
        full_displacement[points_not_inside_trihull, :] = simplified_displacement[bad_index_in_org, :]'''
        with h5py.File('/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/src/displacement_solutions/Y1000000.0_P0.49_23p/'
                       + str(i+46).zfill(3) + '.h5', 'w') as hf:
            hf.create_dataset("simplified_displacement", data=simplified_displacement)
            hf.create_dataset("displacement", data=full_displacement)
        '''plot_meshes(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
                    simplified_mesh_deformed.vertices, simplified_mesh_deformed.faces, back_index_simplified)
        plot_meshes(mesh.vertices+full_displacement, mesh.faces,
                    mesh.vertices+full_displacement, mesh.faces, back_index)
        plot_meshes(mesh.vertices+full_displacement, mesh.faces,
                    simplified_mesh_deformed.vertices, simplified_mesh_deformed.faces, back_index)'''

def full_displacement():
    mesh_back, boundary_vindice = generate_complement(pymesh.load_mesh('../../org/Liver.off'), pymesh.load_mesh('../../org/Liver_Front.off'))
    #plot_mesh(mesh_back.vertices, mesh_back.faces)
    mean, pca_vectors = linear_PCA(mesh_back.vertices)
    #plot_mesh_surfacenormal(mesh_back.vertices, mesh_back.faces, mean, pca_vectors[1])

    mesh, stiff, laplacian = load_stiff(1.0e6, 0.49)
    back_index = get_back_area(mesh, mesh_back, pymesh.load_mesh('../../org/Liver_Front.off'))

    liver_model = LiverModel()
    #liver_model.build_stiffness()
    triangulation = liver_model.generate_vertices_triangulation()
    back_index_simplified, index_in_org = produce_back_index(back_index, liver_model)

    #displacement, ranges = generate_displacement(mesh, C, pca_vectors)
    with h5py.File('coefficient.h5', 'r') as hf:
        coefficient = hf['coefficient'][:]
        coefficient[4] = 0.0
    displacements = load_disp_solutions('../displacement_solutions/Y1.0e4_P0.4', np.sum(num_of_combination[:(order_of_polynomial+1)]))
    CC = np.random.normal(scale=0.3, size=np.sum(num_of_combination[:(order_of_polynomial + 1)]))
    for i in range(np.sum(num_of_combination[:(order_of_polynomial+1)])):
        C = np.zeros(shape=[np.sum(num_of_combination[:(order_of_polynomial+1)])])
        C[i] = 1.0
        C = 0.05 * C * coefficient
        C[4] = 0.0
        displacement, ranges = generate_displacement(mesh, C, mean, pca_vectors)

        approxi_displacement = np.reshape(np.matmul(displacements, CC), [-1, 3])
        #mesh_deformed2 = pymesh.form_mesh(mesh.vertices+approxi_displacement, mesh.faces)
        #plot_meshes(mesh.vertices, mesh.faces,
        #            mesh_deformed2.vertices, mesh_deformed2.faces, back_index)
        #plot_mesh_points(mesh_deformed2.vertices, mesh_deformed2.faces, mesh.vertices+displacement)
        print('volume: ' + str(mesh.volume))
        simplified_displacement = displacement[index_in_org]

        '''plot_meshes(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
                    liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces, back_index_simplified)
        plot_meshes(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
                    liver_model.simplified_tetr_mesh.vertices+simplified_displacement, liver_model.simplified_tetr_mesh.faces, back_index_simplified)'''
        with h5py.File('/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/src/displacement_solutions/Y1000000.0_P0.49/'
                       + str(i).zfill(3) + '.h5', 'r') as hf:
            simplified_displacement = hf["simplified_displacement"][:]
        from util_package.util import tetra_interpolation
        full_displacement, points_not_inside_trihull = tetra_interpolation(mesh.vertices,
                                                liver_model.simplified_tetr_mesh.vertices,
                                                simplified_displacement,
                                                triangulation)
        distance = np.expand_dims(liver_model.simplified_tetr_mesh.vertices, axis=1) - np.expand_dims(
            mesh.vertices[points_not_inside_trihull, :], axis=0)
        distance = np.sum(distance * distance, axis=2)
        index_in_org = np.argmin(distance, axis=0)
        full_displacement[points_not_inside_trihull, :] = simplified_displacement[index_in_org, :]

        #with h5py.File('/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/src/displacement_solutions/Y1000000.0_P0.49/'
        #               + str(i).zfill(3) + '.h5', 'w') as hf:
        #    hf.create_dataset("displacement", data=full_displacement)

        plot_meshes(liver_model.simplified_tetr_mesh.vertices + simplified_displacement, liver_model.simplified_tetr_mesh.faces,
                    mesh.vertices + full_displacement, mesh.faces, back_index_simplified)
        '''plot_meshes(liver_model.simplified_tetr_mesh.vertices, liver_model.simplified_tetr_mesh.faces,
                    simplified_mesh_deformed.vertices, simplified_mesh_deformed.faces, back_index_simplified)
        plot_meshes(mesh.vertices+full_displacement, mesh.faces,
                    simplified_mesh_deformed.vertices, simplified_mesh_deformed.faces, back_index)'''


if __name__ == "__main__":
    main()



























