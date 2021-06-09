import pymesh
import numpy as np
from pymesh import Material
import h5py
import os
import scipy
import trimesh
from scipy.sparse.linalg import inv
from scipy.linalg import pinv
import numpy

from util_package.util import load_PC, parse, angle2rotmatrix
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def append_stiffness_matrix(stiffness_matrix):
    num_nodes = stiffness_matrix.shape[0]/3
    one = np.eye(3)
    ones = np.tile(np.expand_dims(one, axis=0), [int(num_nodes), 1, 1])
    ones = np.reshape(np.transpose(ones, [1,0,2]), [3, -1])
    ones_sparse = scipy.sparse.csr_matrix(ones)
    extended = scipy.sparse.vstack([stiffness_matrix, ones_sparse], format='csc')
    return extended

class LiverModel():
    def __init__(self):
        # vertices, faces, voxels, tetr_mesh
        self.load_mesh()
        # face_label: Front:1 FF:2 LR:3 RR:4
        self.load_face_label()
        # surface_nodes, surface_nodes_orgindex, node_is_onsurface,
        self.extract_surface_nodes()
        # simplified_surface_mesh, simplified_face_label, simplified_tetr_mesh
        self.simplify_mesh()
        # stiffness, stiffness_tall, simplified_stiffness
        self.stiffness = 0
        #self.build_stiffness()

    def load_mesh(self):
        h5f = h5py.File(ROOT_DIR+'/deform_liver/tetr_mesh.h5', 'r')
        self.vertices = h5f['mesh_vertices'][:]
        self.faces = h5f['mesh_faces'][:]
        self.voxels = h5f['mesh_voxels'][:]
        # inv_stiff = h5f['inv_stiffness'][:]
        # stiff = h5f['stiffness'][:]
        h5f.close()
        self.tetr_mesh = pymesh.form_mesh(self.vertices, self.faces, self.voxels)

    def simplify_mesh(self):
        #simplified_vertices, simplified_faces, info = pymesh.collapse_short_edges_raw(self.vertices, self.faces, rel_threshold=1.2)
        #self.simplified_surface_mesh = pymesh.form_mesh(simplified_vertices, simplified_faces)
        #self.simplified_face_label = self.face_label[info['source_face_index']]
        self.simplified_surface_mesh = self.surface_mesh
        #self.simplified_tetr_mesh = pymesh.tetrahedralize(self.simplified_surface_mesh, cell_size=0.01, engine='tetgen')
        print(' ')

        tetgen = pymesh.tetgen()
        tetgen.points = self.surface_mesh.vertices
        tetgen.triangles = self.surface_mesh.faces
        #tetgen.tetrahedra = self.voxels
        tetgen.max_tet_volume = 0.0000002
        tetgen.split_boundary = False
        tetgen.run()
        self.simplified_tetr_mesh = tetgen.mesh

        self.simplified_mesh = {}
        simplified_tetr_mesh = pymesh.form_mesh(self.simplified_tetr_mesh.vertices, self.simplified_tetr_mesh.faces)
        meshS, info = pymesh.remove_isolated_vertices(simplified_tetr_mesh)
        self.simplified_mesh['surface_mesh'] = meshS
        self.simplified_mesh['surface_nodes_orgindex'] = info['ori_vertex_index']
        #from util_package.plot import plot_mesh
        #plot_mesh(self.simplified_tetr_mesh.vertices, self.simplified_tetr_mesh.faces)


    def load_face_label(self):
        with h5py.File(ROOT_DIR+'/deform_liver/new_face_label.h5', 'r') as f:
            self.face_label = f['face_label'][:]

    def get_stiffness(self):
        if isinstance(self.stiffness, scipy.sparse.csc_matrix):
            return self.stiffness
        else:
            self.build_stiffness()
            return self.stiffness

    def build_stiffness(self, young=1.0e6, poisson=0.49):
        mat = Material.create_isotropic(3, 1.0, young, poisson)
        assembler = pymesh.Assembler(self.tetr_mesh, material=mat)
        self.stiffness = assembler.assemble("stiffness")

        mat = Material.create_isotropic(3, 1.0, young, poisson)
        assembler = pymesh.Assembler(self.simplified_tetr_mesh, material=mat)
        self.simplified_stiffness = assembler.assemble("stiffness")
        print(' ')
        '''a = assembler.assemble("mass")
        b = assembler.assemble("lumped_mass")
        c = assembler.assemble("laplacian")
        d = assembler.assemble("displacement_strain")
        e = assembler.assemble("elasticity_tensor")
        f = assembler.assemble("engineer_strain_stress")
        g = assembler.assemble("rigid_motion")
        h = assembler.assemble("gradient")
        i = assembler.assemble("graph_laplacian")
        self.stiffness_tall = append_stiffness_matrix(self.stiffness)'''

        #stiffness_eye = self.stiffness + scipy.sparse.eye(self.stiffness.shape[0], format='csc')
        #inv_stiffness = inv(stiffness_eye)
        #print('done')
        #self.inv_stiffness = inv(stiffness_tall.T.dot(stiffness_tall))

    def build_laplacian(self):
        assembler = pymesh.Assembler(self.surface_mesh)
        self.laplacian = assembler.assemble("laplacian")

    def build_laplacian_volume(self):
        assembler = pymesh.Assembler(self.tetr_mesh)
        self.laplacian_volume = assembler.assemble("laplacian")
        return self.laplacian_volume

    def build_laplacian_simplified(self):
        assembler = pymesh.Assembler(self.simplified_tetr_mesh)
        self.laplacian_simplified = assembler.assemble("laplacian")
        return self.laplacian_simplified

    def extract_surface_nodes(self):
        meshS = pymesh.form_mesh(self.vertices, self.faces)
        meshS, info = pymesh.remove_isolated_vertices(meshS)
        meshS.add_attribute('vertex_normal')
        self.surface_mesh = meshS
        self.surface_mesh_vertices_normal = meshS.get_attribute('vertex_normal')
        self.surface_nodes = meshS.vertices
        self.surface_nodes_orgindex = info['ori_vertex_index']
        node_is_onsurface = np.zeros(shape=(self.vertices.shape[0]), dtype=np.int8)
        node_is_onsurface[self.surface_nodes_orgindex] = 1
        self.node_is_onsurface = node_is_onsurface

        nodes_label = np.zeros(shape=[self.vertices.shape[0]], dtype=np.int8)
        sub_structure_nodes_index = np.reshape(self.faces[np.where(self.face_label==1)[0], :], [-1])
        nodes_label[sub_structure_nodes_index] = 1
        sub_structure_nodes_index = np.reshape(self.faces[np.where(self.face_label==2)[0], :], [-1])
        nodes_label[sub_structure_nodes_index] = 2
        sub_structure_nodes_index = np.reshape(self.faces[np.where(self.face_label==3)[0], :], [-1])
        nodes_label[sub_structure_nodes_index] = 3
        sub_structure_nodes_index = np.reshape(self.faces[np.where(self.face_label==4)[0], :], [-1])
        nodes_label[sub_structure_nodes_index] = 4
        self.surface_nodes_label = nodes_label[self.surface_nodes_orgindex]

        meshS.add_attribute('vertex_area')
        self.surface_nodes_mass = meshS.get_attribute('vertex_area')

        meshS.add_attribute('vertex_normal')
        self.surface_nodes_normal = np.reshape(meshS.get_attribute('vertex_normal'), [-1, 3])



    def shift_n_scale(self, points):
        output = self.scale * (points-self.center)
        return output

    def set_scale(self, scale):
        max = np.max(self.surface_mesh.vertices, axis=0)
        min = np.min(self.surface_mesh.vertices, axis=0)
        self.center = (max + min) / 2
        self.scale = scale
        return self.scale, self.center

    def generate_vertices_triangulation(self, young=1.0e6, poisson=0.49):
        self.build_stiffness(young, poisson)
        from scipy.spatial import Voronoi, Delaunay
        self.smp_vert_triangulation = Delaunay(self.simplified_tetr_mesh.vertices)
        return self.smp_vert_triangulation

def get_closest_faces(points, tran_v, liver_model):
    mesh = trimesh.Trimesh(tran_v, liver_model.faces, process=False)
    closest_p, distance, face_id = trimesh.proximity.closest_point_naive(mesh, points)
    return face_id

def create_face_label():
    liver_model = LiverModel()
    new_face_label = liver_model.face_label
    new_face_label[np.where(new_face_label == 1)] = 1
    new_face_label[np.where(new_face_label == 2)] = 1
    new_face_label[np.where(new_face_label == 3)] = 1
    new_face_label[np.where(new_face_label == 4)] = 1

    '''good_regs2 = [1, 2, 3, 4, 38]
    for i in good_regs2:
        with h5py.File('../../org/reg_dataset/Set' + str(i).zfill(3) + '/rigid.h5', 'r') as f:
            error = f['error'].value
            print(error)
            parameters = f['parameters'][:]
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R = parse(parameters)
        tran_v, rotation_matrix = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)

        face_index = get_closest_faces(point_cloud[np.where(PC_label==2)[0], :], tran_v, liver_model)
        new_face_label[face_index] = 2

    good_regs3 = [1, 2, 3, 4, 38]
    for i in good_regs3:
        with h5py.File('../../org/reg_dataset/Set' + str(i).zfill(3) + '/rigid.h5', 'r') as f:
            error = f['error'].value
            print(error)
            parameters = f['parameters'][:]
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R = parse(parameters)
        tran_v, rotation_matrix = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)
        face_index = get_closest_faces(point_cloud[np.where(PC_label == 3)[0], :], tran_v, liver_model)
        new_face_label[face_index] = 3

    good_regs4 = [2, 13, 23, 38]
    for i in good_regs4:
        with h5py.File('../../org/reg_dataset/Set' + str(i).zfill(3) + '/rigid.h5', 'r') as f:
            error = f['error'].value
            print(error)
            parameters = f['parameters'][:]
        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
        point_cloud = point_cloud / 1000

        S, T, R = parse(parameters)
        tran_v, rotation_matrix = apply_transform(S, T, R, liver_model.tetr_mesh.vertices)
        face_index = get_closest_faces(point_cloud[np.where(PC_label == 4)[0], :], tran_v, liver_model)
        new_face_label[face_index] = 4'''

    face_index = get_face_index('../../org/Liver_FFc_new.off', liver_model)
    new_face_label[face_index] = 2
    face_index = get_face_index('../../org/Liver_LRc_new.off', liver_model)
    new_face_label[face_index] = 3
    face_index = get_face_index('../../org/Liver_RRc_new.off', liver_model)
    new_face_label[face_index] = 4

    from util_package.plot import plot_meshes
    plot_meshes(liver_model.vertices, liver_model.faces,
                liver_model.vertices, liver_model.faces[np.where(new_face_label>=2)[0]])
    with h5py.File(ROOT_DIR + '/deform_liver/new_face_label.h5', 'w') as f:
        f.create_dataset('face_label', data=new_face_label)

def sparse_pinv(left_Matrix_s):
    u, s, v = scipy.sparse.linalg.svds(left_Matrix_s, k=5)
    s_inv = np.diag(1/s)
    K11E_inv_svd = np.matmul(v.T, np.matmul(s_inv, u.T))
    return K11E_inv_svd

def get_face_index(filename, liver_model):
    new_RR = pymesh.load_mesh(filename)
    non_RR_faces = new_RR.faces
    with h5py.File('../V_order.h5', 'r') as f:
        v_order = f['order'][:]
        non_RR_faces_old = v_order[non_RR_faces]

    non_RR_faces_old_list = non_RR_faces_old.tolist()
    is_non_RR = []
    for i in liver_model.faces.tolist():
        is_non_RR.append(i in non_RR_faces_old_list)
    face_index = np.where(np.asarray(is_non_RR) == False)
    return face_index

def debug():
    livermodel = LiverModel()
    mesh_tri = trimesh.Trimesh(livermodel.simplified_tetr_mesh.vertices, livermodel.simplified_tetr_mesh.faces, process=False)
    face_normals = mesh_tri.face_normals[np.arange(0, livermodel.simplified_tetr_mesh.faces.shape[0]),:]
    face_centers = livermodel.simplified_tetr_mesh.vertices[livermodel.simplified_tetr_mesh.faces[:,0], :] \
                   + livermodel.simplified_tetr_mesh.vertices[livermodel.simplified_tetr_mesh.faces[:,1], :] \
                   + livermodel.simplified_tetr_mesh.vertices[livermodel.simplified_tetr_mesh.faces[:,2], :]
    face_centers = face_centers / 3

    from util_package.plot import plot_mesh_vectors
    plot_mesh_vectors(livermodel.simplified_tetr_mesh.vertices, livermodel.simplified_tetr_mesh.faces,
                      face_centers, face_normals)

def debug_laplacian():
    from util_package.plot import plot_mesh_points, plot_meshes_n_points

    liver_model = LiverModel()
    liver_model.build_stiffness()
    L = liver_model.build_laplacian_volume()

    surface_boundary_vetices_index = np.where(liver_model.surface_nodes_label >= 1)[0]
    boundary_vetices_index = liver_model.surface_nodes_orgindex[surface_boundary_vetices_index]
    unknow_vetices_indicator = np.zeros(shape=liver_model.vertices.shape[0])
    unknow_vetices_indicator[boundary_vetices_index] = 1
    unknow_vetices_index = np.where(unknow_vetices_indicator==0)[0]

    displacement = np.zeros_like(liver_model.vertices)
    displacement[liver_model.surface_nodes_orgindex[np.where(liver_model.surface_nodes_label == 2)[0]], 0] = 0.1
    displacement[liver_model.surface_nodes_orgindex[np.where(liver_model.surface_nodes_label == 3)[0]], 1] = 0.1
    displacement[liver_model.surface_nodes_orgindex[np.where(liver_model.surface_nodes_label == 4)[0]], 2] = 0.1
    plot_meshes_n_points(liver_model.vertices, liver_model.tetr_mesh.faces, \
                         liver_model.vertices+displacement, liver_model.tetr_mesh.faces, \
                         liver_model.vertices+displacement)

    solver = pymesh.SparseSolver.create('LDLT')
    solver.compute(liver_model.stiffness)
    LL = L[unknow_vetices_index, :]
    Li = LL[:, unknow_vetices_index]
    Lb = LL[:, boundary_vetices_index]

    unknown1 = scipy.sparse.linalg.spsolve(Li,
                                -Lb.dot(displacement[boundary_vetices_index, :]))
    solver = pymesh.SparseSolver.create('LDLT')
    solver.compute(Li)
    unknown2 = solver.solve(-Lb.dot(displacement[boundary_vetices_index, :]))
    displacement[unknow_vetices_index, :] = unknown2
    plot_meshes_n_points(liver_model.vertices, liver_model.tetr_mesh.faces, \
                         liver_model.vertices + displacement, liver_model.tetr_mesh.faces, \
                         liver_model.vertices + displacement)
    s = plot_meshes_n_points(liver_model.vertices, liver_model.tetr_mesh.faces, \
                         liver_model.vertices + displacement, liver_model.tetr_mesh.faces, \
                         liver_model.vertices + displacement, show=False)
    stiffness_symmetric = (liver_model.stiffness.T + liver_model.stiffness) / 2
    from mayavi import mlab
    @mlab.animate(delay=50, ui=True)
    def anim(displacement):
        for i in range(1000):
            displacement_demaen = displacement - np.mean(displacement, axis=0, keepdims=True)
            dEdd = np.reshape(stiffness_symmetric.dot(np.reshape(displacement_demaen, [-1])), [-1, 3])
            '''max = np.max(dEdd)
            dEdd = dEdd / np.max(dEdd)
            dEdd = np.tanh(3 * dEdd)
            dEdd = dEdd * max'''
            displacement -= 1e-6 * dEdd
            nodes = liver_model.vertices + displacement
            s.mlab_source.x = nodes[:, 0]
            s.mlab_source.y = nodes[:, 2]
            s.mlab_source.z = nodes[:, 1]
            plot_meshes_n_points(liver_model.vertices, liver_model.tetr_mesh.faces, \
                                 liver_model.vertices + displacement, liver_model.tetr_mesh.faces, \
                                 liver_model.vertices + displacement)
            yield

    anim(displacement)
    mlab.show()

def debug_laplacian_simplify():
    from util_package.plot import plot_mesh_points, plot_meshes_n_points

    liver_model = LiverModel()
    liver_model.build_stiffness()
    stiffness_inv = sparse_pinv(liver_model.simplified_stiffness)
    L = liver_model.build_laplacian_simplified()

    surface_boundary_vetices_index = np.where(liver_model.surface_nodes_label >= 1)[0]
    boundary_vetices_index = liver_model.simplified_mesh['surface_nodes_orgindex'][surface_boundary_vetices_index]
    unknow_vetices_indicator = np.zeros(shape=liver_model.simplified_tetr_mesh.vertices.shape[0])
    unknow_vetices_indicator[boundary_vetices_index] = 1
    unknow_vetices_index = np.where(unknow_vetices_indicator==0)[0]

    displacement = np.zeros_like(liver_model.simplified_tetr_mesh.vertices)
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 2)[0]], 0] = 0.1
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 3)[0]], 1] = 0.2
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 4)[0]], 2] = 0.1
    #plot_meshes_n_points(liver_model.vertices, liver_model.tetr_mesh.faces, \
    #                     liver_model.simplified_tetr_mesh.vertices+displacement, liver_model.simplified_tetr_mesh.faces, \
    #                     liver_model.simplified_tetr_mesh.vertices+displacement)
    force = displacement

    LL = L[unknow_vetices_index, :]
    Li = L[:, unknow_vetices_index]
    Lb = L[:, boundary_vetices_index]

    solver = pymesh.SparseSolver.create('SparseQR')
    solver.compute(Li)
    unknown = solver.solve(np.reshape(displacement, [-1]))
    unknown = 0.1*(unknown/np.max(unknown))
    displacement1 = np.reshape(unknown, [-1, 3])

    displacement = np.zeros_like(liver_model.simplified_tetr_mesh.vertices)
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 2)[0]], 0] = 0.1
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 3)[0]], 1] = 0.1
    displacement[liver_model.simplified_mesh['surface_nodes_orgindex'][np.where(liver_model.surface_nodes_label == 4)[0]], 2] = 0.2
    unknown = solver.solve(np.reshape(displacement, [-1]))
    unknown = 0.1*(unknown/np.max(unknown))
    displacement2 = np.reshape(unknown, [-1, 3])
    #unknown = scipy.sparse.linalg.spsolve(Li,-Lb.dot(displacement[boundary_vetices_index, :]))
    #displacement[unknow_vetices_index, :] = unknown
    plot_meshes_n_points(liver_model.simplified_tetr_mesh.vertices + displacement1,
                             liver_model.simplified_tetr_mesh.faces, \
                             liver_model.simplified_tetr_mesh.vertices + displacement2,
                             liver_model.simplified_tetr_mesh.faces, \
                             liver_model.simplified_tetr_mesh.vertices + displacement  )
    from mayavi import mlab
    @mlab.animate(delay=50, ui=True)
    def anim(displacement):
        for i in range(1000):
            #dEdd = np.matmul(np.reshape(displacement, [-1]), (liver_model.simplified_stiffness+liver_model.simplified_stiffness.T))
            dEdd = np.matmul(np.expand_dims(np.reshape(displacement, [-1]), axis=1).T,
                      (liver_model.simplified_stiffness + liver_model.simplified_stiffness.T).todense())
            dEdd = np.reshape(dEdd, [-1, 3])
            displacement -= 2e-7 * dEdd
            nodes = liver_model.simplified_tetr_mesh.vertices + displacement
            s.mlab_source.x = nodes[:, 0]
            s.mlab_source.y = nodes[:, 2]
            s.mlab_source.z = nodes[:, 1]
            yield

    anim(displacement)
    mlab.show()

if __name__ == '__main__':
    create_face_label()
    debug_laplacian()
# debug()
    Li_pinv = sparse_pinv(Li)
    diff = np.asarray(np.matmul(Li.todense(), np.matmul(Li_pinv, Li.todense()))) - Li

    unknown = np.matmul(Li_pinv, -Lb.dot(liver_model.vertices[boundary_vetices_index, :]))
    #livermodel.simplify_mesh()

    #a =livermodel.get_stiffness()

    #create_face_label()
    print(' ')