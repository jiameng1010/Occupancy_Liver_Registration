import h5py
import numpy as np
import trimesh

h5f = h5py.File('./LM_fit/tetr_mesh.h5', 'r')
vertices = h5f['mesh_vertices'][:]
faces = h5f['mesh_faces'][:]
voxels = h5f['mesh_voxels'][:]
# inv_stiff = h5f['inv_stiffness'][:]
# stiff = h5f['stiffness'][:]
h5f.close()

mesh_liver = trimesh.load('../org/Liver.off')
distance = np.expand_dims(vertices, axis=0) - np.expand_dims(mesh_liver.vertices, axis=1)
distance = np.sum(distance * distance, axis=2)
order = np.argmin(distance, axis=0)

print(' ')

