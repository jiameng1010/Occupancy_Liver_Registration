import pymesh
import h5py
import numpy as np
import trimesh
from mayavi import mlab
from pymesh import Material

def stiffness_matrix():
    mesh = pymesh.load_mesh('../org/Liver.off')
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    tetr_mesh = pymesh.tetrahedralize(mesh, cell_size=0.007, engine='tetgen')
    print(tetr_mesh.voxels.shape)

    young = 1000.0
    poisson = 0.3
    mat = Material.create_isotropic(3, 1.0, young, poisson)
    assembler = pymesh.Assembler(tetr_mesh, material=mat)
    Stiff = assembler.assemble("stiffness")
    print(' ')

def generate():
    mesh = pymesh.load_mesh('../org/Liver.off')
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    tetr_mesh = pymesh.tetrahedralize(mesh, cell_size=0.007, engine='tetgen')
    print(tetr_mesh.voxel.shape)
    f = h5py.File('../org/Liver_less_tetr.hdf5', 'w')
    f.create_dataset('vertices', data=tetr_mesh.vertices)
    f.create_dataset('faces', data=tetr_mesh.faces)
    f.create_dataset('voxels', data=tetr_mesh.voxels)
    f.close()
    print(' ')

def generate_front():
    mesh = pymesh.load_mesh('../org/Liver_FF.off')
    mesh, _ = pymesh.remove_isolated_vertices(mesh)

    f = h5py.File('../org/Liver_FF.hdf5', 'w')
    f.create_dataset('vertices', data=mesh.vertices)
    f.create_dataset('faces', data=mesh.faces)
    f.close()
    print(' ')

def view():
    ff = h5py.File('../org/Liver_less_tetr.hdf5', 'r')
    tet_v = ff['vertices'][:]
    tet_f = ff['faces'][:]
    tet_t = ff['voxels'][:]
    ff.close()

    mean_v = np.mean(tet_v, axis=0)
    tet_v_mask = np.where(tet_v[:, 0] > mean_v[0])
    v_v = tet_v[tet_v_mask]
    f_back_mask = np.isin(tet_t[:, 0], tet_v_mask)
    f_back_mask = np.logical_or(f_back_mask, np.isin(tet_t[:, 1], tet_v_mask))
    f_back_mask = np.logical_or(f_back_mask, np.isin(tet_t[:, 2], tet_v_mask))
    f_back_mask = np.logical_or(f_back_mask, np.isin(tet_t[:, 3], tet_v_mask))
    v_t = tet_t[np.where(f_back_mask == True)]

    t_f1 = np.concatenate([v_t[:, 0], v_t[:, 1], v_t[:, 2]], axis=0)
    t_f1 = np.resize(t_f1, (-1, 3))
    t_f2 = np.concatenate([v_t[:, 0], v_t[:, 3], v_t[:, 2]], axis=0)
    t_f2 = np.resize(t_f2, (-1, 3))
    t_f3 = np.concatenate([v_t[:, 0], v_t[:, 1], v_t[:, 3]], axis=0)
    t_f3 = np.resize(t_f3, (-1, 3))
    t_f4 = np.concatenate([v_t[:, 3], v_t[:, 1], v_t[:, 2]], axis=0)
    t_f4 = np.resize(t_f4, (-1, 3))
    v_f = np.concatenate([t_f1, t_f2, t_f3, t_f4], axis=0)

    random_color = np.random.uniform(size=v_f.shape)
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         v_t[:,0:3],
                         opacity=0.2)
    mlab.show()

def generate_complement():
    mesh_all = pymesh.load_mesh('../org/Liver.off')
    mesh = pymesh.load_mesh('../org/Liver_FFc.off')

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
    pymesh.save_mesh('../org/Liver_FF.off', mesh_c)

    print('Done')

stiffness_matrix()
#generate_complement()
#view()