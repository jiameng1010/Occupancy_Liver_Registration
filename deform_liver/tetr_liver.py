import trimesh
import numpy as np
import copy
import igl

# load the liver mesh and save it to OFF file
node_f = open('../../org/LiverVolume.nod', 'r')
tetr_f = open('../../org/LiverVolume.elm', 'r')
face_f = open('../../org/LiverVolume.bel', 'r')

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

tetrs = []
for t in tetr_str:
    tt = np.array(t.split(' ')).astype(np.int)[1:] - 1
    tetrs.append(tt)
tetrs = np.array(tetrs)[:,:]

mesh_liver = trimesh.Trimesh(vertices=vertices, faces=tetrs)

lf = igl.cotmatrix(vertices, faces)
lt = igl.cotmatrix(vertices, tetrs)
mesh_liver.show()