import igl
import trimesh
import numpy as np
import open3d as o3d
import copy
import pymesh
#from mayavi import mlab
#from functools import reduce
##from scipy import sparse
import h5py

def normalize(normal_vectors):
    norm = np.sum(normal_vectors*normal_vectors, axis=1, keepdims=True)
    return (1.0/norm) * normal_vectors

def mesh_volume(mesh):
    v0 = mesh.vertices[mesh.voxels[:, 0],:]
    v1 = mesh.vertices[mesh.voxels[:, 1], :]
    v2 = mesh.vertices[mesh.voxels[:, 2], :]
    v3 = mesh.vertices[mesh.voxels[:, 3], :]
    tetra_volume = size_tetra(v0, v1, v2, v3)
    return np.sum(tetra_volume)

def size_tetra(v0, v1, v2, v3):
    v11 = v1 - v0
    v22 = v2 - v0
    v33 = v3 - v0
    xa = v11[:, 0]
    ya = v11[:, 1]
    za = v11[:, 2]
    xb = v22[:, 0]
    yb = v22[:, 1]
    zb = v22[:, 2]
    xc = v33[:, 0]
    yc = v33[:, 1]
    zc = v33[:, 2]

    result = xa*(yb*zc - zb*yc) - ya*(xb*zc - zb*xc) + za*(xb*yc - yb*xc)
    return np.abs(result/6)

def parse(para):
    return para[0], para[1:4], para[4:7]

def parse_non_rigid(para):
    return para[0], para[1:4], para[4:7], para[7:]

def angle2rotmatrix(angle):
    r1 = np.asarray([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                     [np.sin(angle[2]), np.cos(angle[2]), 0],
                     [0, 0, 1]])
    r2 = np.asarray([[np.cos(angle[1]), 0, np.sin(angle[1])],
                     [0.0, 1.0, 0.0],
                     [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    r3 = np.asarray([[1, 0, 0],
                     [0, np.cos(angle[0]), -np.sin(angle[0])],
                     [0, np.sin(angle[0]), np.cos(angle[0])]])
    return r1, r2, r3, np.matmul(r1, np.matmul(r2, r3))

def angle2drotmatrix(angle):
    r1 = np.asarray([[-np.sin(angle[2]), -np.cos(angle[2]), 0],
                     [np.cos(angle[2]), -np.sin(angle[2]), 0],
                     [0, 0, 0.0]])
    r2 = np.asarray([[-np.sin(angle[1]), 0, np.cos(angle[1])],
                     [0.0, 0.0, 0.0],
                     [-np.cos(angle[1]), 0, -np.sin(angle[1])]])
    r3 = np.asarray([[0, 0, 0],
                     [0, -np.sin(angle[0]), -np.cos(angle[0])],
                     [0, np.cos(angle[0]), -np.sin(angle[0])]])
    return r1, r2, r3

def load_PC(data_dir):
    FF = data_dir + '/' + data_dir.split('/')[-1] + "FF.xyz"
    LR = data_dir + '/' + data_dir.split('/')[-1] + "LR.xyz"
    RR = data_dir + '/' + data_dir.split('/')[-1] + "RR.xyz"
    SR = data_dir + '/' + data_dir.split('/')[-1] + "SR.xyz"
    FF_f = open(FF, 'r')
    LR_f = open(LR, 'r')
    RR_f = open(RR, 'r')
    SR_f = open(SR, 'r')
    FF_str = FF_f.readlines()
    LR_str = LR_f.readlines()
    RR_str = RR_f.readlines()
    SR_str = SR_f.readlines()
    FF_f.close()
    LR_f.close()
    RR_f.close()
    SR_f.close()

    PC_str = FF_str + LR_str + RR_str + SR_str
    Points = []
    for v in PC_str:
        vv = np.array(v.split(' ')).astype(np.float)[1:]
        Points.append(vv)
    point_cloud = np.array(Points)

    label = np.concatenate([0 * np.ones(shape=len(FF_str)),
                            1 * np.ones(shape=len(LR_str)),
                            2 * np.ones(shape=len(RR_str)),
                            3 * np.ones(shape=len(SR_str)),], axis=0)

    return point_cloud, label

def init_reg(point_cloud, PC_lable, tet_v, meshes):
    target0 = point_cloud[np.where(PC_lable==0)[0], :]
    pc_target0 = o3d.geometry.PointCloud()
    pc_target0.points = o3d.utility.Vector3dVector(target0)
    pc_target0.paint_uniform_color([1,0,0])
    target1 = point_cloud[np.where(PC_lable==1)[0], :]
    pc_target1 = o3d.geometry.PointCloud()
    pc_target1.points = o3d.utility.Vector3dVector(target1)
    pc_target1.paint_uniform_color([0,1,0])
    target2 = point_cloud[np.where(PC_lable==2)[0], :]
    pc_target2 = o3d.geometry.PointCloud()
    pc_target2.points = o3d.utility.Vector3dVector(target2)
    pc_target2.paint_uniform_color([0,0,1])
    pc_target = pc_target0 + pc_target1 + pc_target2
    pc_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=5000))

    source = tet_v[np.unique(np.resize(meshes[0].faces,(-1))), :]
    pc_source0 = o3d.geometry.PointCloud()
    pc_source0.points = o3d.utility.Vector3dVector(source)
    pc_source0.paint_uniform_color([1,0,0])
    source = tet_v[np.unique(np.resize(meshes[1].faces,(-1))), :]
    pc_source1 = o3d.geometry.PointCloud()
    pc_source1.points = o3d.utility.Vector3dVector(source)
    pc_source1.paint_uniform_color([0,1,0])
    source = tet_v[np.unique(np.resize(meshes[2].faces,(-1))), :]
    pc_source2 = o3d.geometry.PointCloud()
    pc_source2.points = o3d.utility.Vector3dVector(source)
    pc_source2.paint_uniform_color([0,0,1])
    pc_source = pc_source0 + pc_source1 + pc_source2
    pc_source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=5000))

    print("Apply point-to-point ICP")
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.registration.registration_colored_icp(
        pc_source, pc_target, 50, trans_init, o3d.registration.ICPConvergenceCriteria(relative_fitness=1,
                                                    relative_rmse=1,
                                                    max_iteration=20))
    '''reg_p2p = o3d.registration.registration_icp(
        pc_target, pc_source, 3, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())'''
    draw_registration_result(pc_source, pc_target, reg_p2p.transformation)
    return np.asarray([1]), reg_p2p.transformation[0:3, 3], reg_p2p.transformation[0:3, 0:3]

def load_voxel_mesh():
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

    voxels = []
    for f in tetr_str:
        ff = np.array(f.split(' ')).astype(np.int)[1:] - 1
        voxels.append(ff)
    voxels = np.array(voxels)

    # mesh_liver = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh_liver.show()
    # mesh_liver.export('../org/Liver.off')

    mesh_liver = pymesh.form_mesh(vertices=vertices, faces=faces, voxels=voxels)
    return mesh_liver

def load_disp_solutions(dir, num):
    solutions = []
    for i in range(num):
        with h5py.File(dir + '/' + str(i).zfill(3) + '.h5', 'r') as hf:
            coefficient = hf['displacement'][:]
            solutions.append(np.expand_dims(coefficient, 1))
    return np.concatenate(solutions, axis=1)


