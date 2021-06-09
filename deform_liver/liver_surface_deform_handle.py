import igl
import trimesh
import numpy as np
import copy
from mayavi import mlab
from functools import reduce
from scipy import sparse

def get_closet(v, f, handles):
    surface_v_index = reduce(np.union1d, f.tolist())
    v_surface = v[surface_v_index,:]
    distance = np.expand_dims(v_surface, axis=1) - np.expand_dims(handles, axis=0)
    distance = distance * distance
    distance = np.sum(distance, axis=2)
    return surface_v_index[np.argmin(distance, axis=0)]


v, f = igl.read_triangle_mesh('../../org/Liver_surface.off')
k = igl.gaussian_curvature(v, f)
v_color = (0.1)*copy.copy(v[:,0]) + copy.copy(v[:,1]) + -(0.05)*copy.copy(v[:,2])
#v_color[np.where(v_color<np.mean(v_color))[0]] = 0.1
#v_color[np.where(v_color>np.mean(v_color))[0]] = 0.2
face_k = v_color[f[:, 0]] + v_color[f[:, 1]] + v_color[f[:, 2]]
face_k = np.tile(np.expand_dims(face_k, axis=1), [1, 3])
tetr_f = open('../../org/LiverVolume.elm', 'r')
tetr_str = tetr_f.readlines()
tetrs = []
for t in tetr_str:
    tt = np.array(t.split(' ')).astype(np.int)[1:] - 1
    tetrs.append(tt)
tetrs = np.array(tetrs)[:,:]

threshold = np.min(v_color) + 0.65*(np.max(v_color) - np.min(v_color))
v_back_index = np.where(v_color > threshold)
f_back_mask = np.isin(f[:, 0], v_back_index)
f_back_mask = np.logical_or(f_back_mask, np.isin(f[:, 1], v_back_index))
f_back_mask = np.logical_or(f_back_mask, np.isin(f[:, 2], v_back_index))
f_back = f[np.where(f_back_mask == True)]

mesh_liver = trimesh.Trimesh(vertices=v, faces=f, face_colors=face_k)
mesh_liver.show()

mesh_liver_back = trimesh.Trimesh(vertices=v, faces=f_back)
mesh_liver_back.show()
U = igl.decimate(v, f_back, 100)
handles = U[1]
handles_faces = U[2]
handles_index = get_closet(v, f, handles)

mesh_liver1000 = trimesh.Trimesh(vertices=U[1], faces=U[2])
mesh_liver1000.show()
random_displacement = np.zeros_like(handles)
a = 0.2
random_displacement[:, 0] = np.random.normal(scale=a*(np.max(handles[:,0]) - np.min(handles[:,])), size=handles.shape[0])
random_displacement[:, 1] = np.random.normal(scale=a*(np.max(handles[:,0]) - np.min(handles[:,])), size=handles.shape[0])
random_displacement[:, 2] = np.random.normal(scale=a*(np.max(handles[:,0]) - np.min(handles[:,])), size=handles.shape[0])
adjacency_back = igl.adjacency_matrix(U[2])
L_back = adjacency_back.toarray()
L_back = np.diag(np.sum(L_back, axis=0)) + L_back
L_back = L_back / np.sum(L_back, axis=0, keepdims=True)
random_displacement = random_displacement * np.expand_dims(np.sum(L_back, axis=0), axis=1)

mesh_liver1000 = trimesh.Trimesh(vertices=U[1]+random_displacement, faces=U[2])
mesh_liver1000.show()

for i in range(5):
    random_displacement = np.matmul(L_back, random_displacement)
    current = U[1]+random_displacement
mlab.figure(bgcolor=(1, 1, 1))
mlab.triangular_mesh([vert[0] for vert in U[1]],
                     [vert[2] for vert in U[1]],
                     [vert[1] for vert in U[1]],
                     U[2],
                     opacity=0.2,
                     color=(0,1,0))

mlab.triangular_mesh([vert[0] for vert in current],
                     [vert[2] for vert in current],
                     [vert[1] for vert in current],
                     U[2],
                     opacity=0.2,
                     color=(0,0,1))
mlab.show()


d = igl.harmonic_weights(v, f, handles_index, random_displacement, 2)
#l_tetr = igl.cotmatrix(v, f)
#m_tetr = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
#d = igl.harmonic_weights_from_laplacian_and_mass(l_tetr, m_tetr, handles_index, random_displacement, 2)
deformed_v = v + d

mlab.figure(bgcolor=(1, 1, 1))
mlab.triangular_mesh([vert[0] for vert in v],
                     [vert[2] for vert in v],
                     [vert[1] for vert in v],
                     f,
                     opacity=0.3,
                     color=(0,1,0))

mlab.triangular_mesh([vert[0] for vert in deformed_v],
                     [vert[2] for vert in deformed_v],
                     [vert[1] for vert in deformed_v],
                     f,
                     opacity=0.3,
                     color=(0,0,1))
mlab.show()