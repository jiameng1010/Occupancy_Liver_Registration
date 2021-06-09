from mayavi import mlab
from util import load_PC
import numpy as np
import h5py
import multiprocessing
import trimesh
import time
from util import load_PC, parse, angle2rotmatrix, parse_non_rigid, load_disp_solutions
from plot import plot_mesh_points_label

num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

def apply_transform_non_rigid(S, T, R, C, tet_v, templates):
    #apply deformation
    tet_v = tet_v + np.reshape(np.matmul(templates, C), [-1, 3])
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr

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

def re_order_templates():
    templates = load_disp_solutions('../displacement_solutions/Y1.0e4_P0.4', np.sum(num_of_combination[:4]))
    templates = np.reshape(templates, [-1, 3, 14])
    with h5py.File('../V_order.h5', 'r') as hf:
        order = hf['order'][:]
    templates = templates[order, :, :]
    templates = np.reshape(templates, [-1, 14])
    return templates

def main():
    while 1:
        with open('../animation/index', 'r') as tf:
            index = tf.readlines()[0]
        index_last = index
        points, PC_label = load_PC('../../org/datasets/Set' + index)
        points = points - np.mean(points, axis=0, keepdims=True)
        points = points / 1000

        mesh_liver = trimesh.load('../../org/Liver.off')
        tet_v = np.asarray(mesh_liver.vertices)
        tet_v = tet_v - np.mean(tet_v, axis=0, keepdims=True)
        #meshes
        mesh_FF = trimesh.load('../../org/Liver_FF.off')
        mesh_LR = trimesh.load('../../org/Liver_LR.off')
        mesh_RR = trimesh.load('../../org/Liver_RR.off')
        mesh_Front = trimesh.load('../../org/Liver_Front.off')

        meshes = [mesh_FF, mesh_LR, mesh_RR, mesh_Front]
        templates = re_order_templates()
        mlab.figure()

        parameter = load_para()
        S, T, R, C = parse_non_rigid(parameter)
        tran_v, rotation_matrix = apply_transform(S, T, R, tet_v)
        #f = mlab.points3d(tet_v[:,0], tet_v[:, 2], tet_v[:, 1])
        #mlab.show()
        scene_objs = plot_mesh_points_label(tran_v, meshes, points, PC_label)

        result = anim(tet_v, scene_objs, index, templates)
        mlab.show()

def updates(f, points):
    f.mlab_source.x = points[:, 0]
    f.mlab_source.y = points[:, 2]
    f.mlab_source.z = points[:, 1]
    #f.scene.camera.azimuth(10)
    #f.scene.render()

def load_para():
    with h5py.File('data.h5', 'r') as hf:
        coefficient = hf['para'][:]
    return coefficient

@mlab.animate(delay=200, ui=False)
def anim(tet_v, objs, index_last, templates):
    '''while points.empty is False:
        p = points.get()
        f = mlab.points3d(p[:,0], p[:,2], p[:,1])
    #mlab.show()'''
    #f = mlab.points3d(tet_v[:,0], tet_v[:, 2], tet_v[:, 1])
    #mlab.show()
    while 1:
        print('**********')
        try:
            parameter = load_para()
        except:
            continue
        #points = points+np.random.normal(0.0, 10, size=points.shape)
        time.sleep(0.1)
        S, T, R, C = parse_non_rigid(parameter)
        tran_v, rotation_matrix = apply_transform_non_rigid(S, T, R, C, tet_v, templates)
        tran_v = tran_v
        #f.mlab_source.x = [vert[0] for vert in tran_v]
        #f.mlab_source.y = [vert[2] for vert in tran_v]
        #f.mlab_source.z = [vert[1] for vert in tran_v]
        objs[4].mlab_source.x = [vert[0] for vert in tran_v]
        objs[4].mlab_source.y = [vert[2] for vert in tran_v]
        objs[4].mlab_source.z = [vert[1] for vert in tran_v]
        objs[5].mlab_source.x = [vert[0] for vert in tran_v]
        objs[5].mlab_source.y = [vert[2] for vert in tran_v]
        objs[5].mlab_source.z = [vert[1] for vert in tran_v]
        objs[6].mlab_source.x = [vert[0] for vert in tran_v]
        objs[6].mlab_source.y = [vert[2] for vert in tran_v]
        objs[6].mlab_source.z = [vert[1] for vert in tran_v]
        objs[7].mlab_source.x = [vert[0] for vert in tran_v]
        objs[7].mlab_source.y = [vert[2] for vert in tran_v]
        objs[7].mlab_source.z = [vert[1] for vert in tran_v]

        with open('../animation/index', 'r') as tf:
            index = tf.readlines()[0]
        if index != index_last:
            break
            '''points_FF = points[np.where(PC_label == 0), :][0]
            points_LR = points[np.where(PC_label == 1), :][0]
            points_RR = points[np.where(PC_label == 2), :][0]
            points_SR = points[np.where(PC_label == 3), :][0]
            objs[0].mlab_source.x = points_FF[:,0]
            objs[0].mlab_source.y = points_FF[:,2]
            objs[0].mlab_source.z = points_FF[:,1]
            objs[1].mlab_source.x = points_LR[:,0]
            objs[1].mlab_source.y = points_LR[:,2]
            objs[1].mlab_source.z = points_LR[:,1]
            objs[2].mlab_source.x = points_RR[:,0]
            objs[2].mlab_source.y = points_RR[:,2]
            objs[2].mlab_source.z = points_RR[:,1]
            objs[3].mlab_source.x = points_SR[:,0]
            objs[3].mlab_source.y = points_SR[:,2]
            objs[3].mlab_source.z = points_SR[:,1]'''

        #updates(f, tet_v)
        yield
    mlab.close()

main()