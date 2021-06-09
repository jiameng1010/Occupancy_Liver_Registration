import trimesh
import pymesh
import numpy as np
import copy
import igl
import h5py

# load the liver mesh and save it to OFF file
node_f = open('../org/LiverVolume.nod', 'r')
tetr_f = open('../org/LiverVolume.elm', 'r')
face_f = open('../org/LiverVolume.bel', 'r')

vertices_str = node_f.readlines()
faces_str = face_f.readlines()
tetr_str = tetr_f.readlines()
vertices = []
for v in vertices_str:
    vv = np.array(v.split(' ')).astype(np.float)[1:]
    vertices.append(vv)
vertices = np.array(vertices)

faces = []
for f in faces_str:
    ff = np.array(f.split(' ')).astype(np.int)[1:] - 1
    faces.append(ff)
faces = np.array(faces)
tmp = copy.copy(faces[:, 0])
faces[:, 0] = faces[:, 1]
faces[:, 1] = tmp

voxels = []
for f in tetr_str:
    ff = np.array(f.split(' ')).astype(np.int)[1:] - 1
    voxels.append(ff)
voxels = np.array(voxels)

#mesh_liver = trimesh.Trimesh(vertices=vertices, faces=faces)
#mesh_liver.show()
#mesh_liver.export('../org/Liver.off')

mesh_liver = pymesh.form_mesh(vertices=vertices, faces=faces, voxels=voxels)
#pymesh.save_mesh('../org/Liver_voxel.off', mesh_liver)
#mesh_liver.show()
#mesh_liver.export('../org/Liver.off')
h5f = h5py.File('./LM_fit/tetr_mesh.h5', 'w')
h5f.create_dataset('mesh_vertices', data=vertices)
h5f.create_dataset('mesh_faces', data=faces)
h5f.create_dataset('mesh_voxels', data=voxels)
h5f.close()