import os
import sys
import subprocess
import socket
import yaml
import h5py
import numpy as np
import trimesh
import pymesh
from trimesh import sample
from scipy.sparse.linalg import spsolve, lsmr
import multiprocessing
import time

hostname = socket.gethostname()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from util_package.util import normalize, parse, apply_rigid_transform,\
    angle2rotmatrix, angle2drotmatrix
if hostname != 'monavale':
    from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, \
        plot_meshes, plot_mesh_force, compute_n_plot_force, plot_mesh_points_vector, plot_meshes_n_points
    from mayavi import mlab
else:
    OCC_FUN_DIR = '/tmp/pycharm_pcnn_liver/normal_prediction'
    sys.path.append(OCC_FUN_DIR)
    from Occupancy_Function import OccFunction
from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point_old

exp_out_dir = '/tmp/exp_out'

class Occ_Reg():
    def __init__(self, points, liver_model, occ_function):
        self.pc = points
        self.point_cloud = points[0]
        self.point_label = points[1]
        self.transformation = points[-1]
        self.GT_mesh = points[-2]
        self.liver_model = liver_model
        self.liver_model.build_stiffness()
        self.liver_model.build_laplacian()
        self.occ_function = occ_function
        self.GT_displacement = points[-2].vertices - self.liver_model.tetr_mesh.vertices

        self.occ_function.update_PC(self.point_cloud)
        self.reconstructied_mesh = self.occ_function.get_reconstructied_mesh()
        self.smoother = LaplacianSmoothing(self.liver_model.laplacian, 100)
        # initialization
        self.transformed_vertices = self.transform(self.liver_model.tetr_mesh.vertices)
        self.transformed_surface_vertices = self.transform(self.liver_model.surface_mesh.vertices)
        self.displacement = np.zeros_like(self.liver_model.tetr_mesh.vertices)
        self.surface_nodes_displacement = np.zeros_like(self.liver_model.surface_mesh.vertices)
        self.initilze_parameters()

    def save_debug(self, filename):
        if not os.path.isfile(exp_out_dir + '/reconstructed_mesh.off'):
            pymesh.save_mesh(exp_out_dir + '/reconstructed_mesh.off', self.reconstructied_mesh)
        with h5py.File(exp_out_dir + '/' + filename, 'w') as f:
            nodes_p = self.apply_registration()
            f.create_dataset('nodes', data=(nodes_p))
            f.create_dataset('surface_nodes', data=nodes_p[self.liver_model.surface_nodes_orgindex, :])

    def compute_IOU(self):
        mesh1 = trimesh.Trimesh(self.apply_registration(), self.liver_model.tetr_mesh.faces)
        mesh2 = trimesh.Trimesh(self.GT_mesh.vertices, self.GT_mesh.faces)
        point_set1, _ = sample.sample_surface(mesh1, 10000)
        scale1 = np.max(np.max(mesh1.vertices, axis=0) - np.min(mesh1.vertices, axis=0))
        point_set1 = point_set1 + np.random.normal(loc=0.0, scale=0.04*scale1, size=point_set1.shape)
        point_set2, _ = sample.sample_surface(mesh2, 10000)
        scale2 = np.max(np.max(mesh2.vertices, axis=0) - np.min(mesh2.vertices, axis=0))
        point_set2 = point_set2 + np.random.normal(loc=0.0, scale=0.04*scale2, size=point_set2.shape)
        point_set = np.concatenate([point_set1, point_set2], axis=0)

        in_out1 = 1.0 * mesh1.contains(point_set)
        in_out2 = 1.0 * mesh2.contains(point_set)
        in_out = in_out1 + in_out2
        return np.sum(1.0 * (in_out==2)) / np.sum(1.0 * (in_out>=1))

    def apply_registration(self):
        S, T, R = parse(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr = apply_rigid_transform(S, T, R, node_position_org)
        node_position += self.displacement
        return node_position

    def evaluate(self):
        error = self.apply_registration() - self.GT_mesh.vertices
        mean_square_error = np.mean(error * error)
        print(mean_square_error)
        #iou = self.compute_IOU()
        #print('IOU: ' + str(iou))

    def transform(self, p):
        return np.matmul(self.transformation[0], (p - self.transformation[1]).T).T
    def transform_inv(self, p):
        transformation_inv = np.linalg.inv(self.transformation[0])
        return np.matmul(transformation_inv, p.T).T + self.transformation[1]

    def initilze_parameters(self):
        S = np.random.uniform(0.99, 1.01, size=1)
        T = 0.0 * np.random.normal(size=[3])
        R = np.random.uniform(0.01*np.pi, 0.01*np.pi, size=3)

        S, T, R = Iterative_Closest_Point_old.ICP(self.liver_model, self.pc)
        self.rigid_parameters = np.concatenate([np.asarray([S]), T, R], axis=0)

    def rigid_fit(self):
        S, T, R = parse(self.rigid_parameters)
        node_position_org = self.transformed_surface_vertices
        node_position, rr = apply_rigid_transform(S, T, R, node_position_org)
        node_position += self.surface_nodes_displacement
        pred, pred_value, pred_normal = self.occ_function.eval(node_position)
        #pred_normal = self.smoother.filtering(pred_normal)

        drdT = pred_normal
        dQdS = np.matmul(rr, node_position_org.T).T
        drdS = np.sum(pred_normal * dQdS, axis=1, keepdims=True)
        r1, r2, r3, rr = angle2rotmatrix(R)
        dr1, dr2, dr3 = angle2drotmatrix(R)
        dQdR1 = dr1.dot(r2.dot(r3.dot(node_position_org.T))).T
        drdR1 = np.sum(pred_normal * dQdR1, axis=1, keepdims=True)
        dQdR2 = r1.dot(dr2.dot(r3.dot(node_position_org.T))).T
        drdR2 = np.sum(pred_normal * dQdR2, axis=1, keepdims=True)
        dQdR3 = r1.dot(r2.dot(dr3.dot(node_position_org.T))).T
        drdR3 = np.sum(pred_normal * dQdR3, axis=1, keepdims=True)

        jacobian = np.concatenate([drdS, drdT, drdR3, drdR2, drdR1], axis=1)
        jTj = np.matmul(jacobian.T, jacobian)
        increament = - np.matmul(np.matmul(np.linalg.inv(jTj + 0.5*np.diag(jTj)), jacobian.T), pred_value)
        #increament[0] = 0.0
        self.rigid_parameters += 0.5*increament

    def run(self, num_iter, save_result=False):
        for i in range(num_iter):
            print('********* step: '+str(i) + ' **********')
            if i % 1 == 0:
                self.save_debug(str(i).zfill(5)+'.h5')
            self.evaluate()

            self.rigid_fit()
            nodes_p = self.apply_registration()
            surface_nodes_p = nodes_p[self.liver_model.surface_nodes_orgindex, :]

            pred, pred_value, pred_normal = self.occ_function.eval(surface_nodes_p)
            #surface_node_force = -1.0e-4* np.expand_dims(np.tanh(3*pred_value), axis=1) * pred_normal
            surface_node_force = -3.0e-5 * np.expand_dims(pred_value, axis=1) * pred_normal #(make sure hav negetive sign)
            if i < 20:
                surface_node_force_smoothed = np.zeros_like(surface_node_force)#self.smoother.filtering(surface_node_force)
            else:
                surface_node_force_smoothed = np.zeros_like(surface_node_force)
            #surface_node_force_smoothed = np.zeros_like(surface_node_force)
            node_force = np.zeros_like(self.displacement)
            node_force[self.liver_model.surface_nodes_orgindex, :] = surface_node_force_smoothed

            self.displacement += node_force
            self.surface_nodes_displacement = self.displacement[self.liver_model.surface_nodes_orgindex, :]

    def solve_equ(self, node_force):
        node_force = np.reshape(node_force, [-1])
        #displacement = lsmr(self.liver_model.stiffness_tall, np.concatenate([node_force, np.asarray([0.0, 0.0, 0.0])], axis=0))
        displacement = lsmr(self.liver_model.stiffness, node_force)
        return np.reshape(displacement[0], [-1, 3])


def main():
    subprocess.run(['rm', '-r', exp_out_dir])
    subprocess.run(['mkdir', exp_out_dir])
    with open('reg_cfg.yaml', 'r') as f:
        cfg = yaml.load(f)

    data_loader = ProbeProvider(cfg)
    pc = data_loader.load_one_pointcloud(False, 6)
    liver_model = LiverModel()
    occ_function = OccFunction(
        '/tmp/pycharm_pcnn_liver/normal_prediction/train_results/2020_07_09_17_37_07/trained_models/model.ckpt')
    registration = Occ_Reg(pc, liver_model, occ_function)
    registration.run(100, save_result=True)




    print('finished')



def get_queue():
    queue = [multiprocessing.Queue(),
             multiprocessing.Queue()]
    return queue

def plot_result():
    with open('debug.yaml', 'r') as f:
        cfg = yaml.load(f)

    data_loader = ProbeProvider(cfg)
    pc = data_loader.load_one_pointcloud(False, 6)
    liver_model = LiverModel()

    filename = str(0).zfill(5) + '.h5'
    reconstructed_mesh = pymesh.load_mesh(exp_out_dir + '/reconstructed_mesh.off')
    with h5py.File(exp_out_dir + '/' + filename, 'r') as f:
        nodes = f['nodes'][:]
        surface_nodes = f['surface_nodes'][:]
        s = plot_meshes_n_points(reconstructed_mesh.vertices, reconstructed_mesh.faces,#np.matmul(pc[-1][0], (pc[-2].vertices - pc[-1][1]).T).T, pc[-2].faces,
                                 nodes, liver_model.tetr_mesh.faces,
                                 pc[0][:pc[-3], :], show=False)

    @mlab.animate
    def anim():
        for i in range(1, 100):
            #if i % 1 != 0:
            #    continue
            #if i == 2:
            #    a = input()
            filename = str(i).zfill(5) + '.h5'
            with h5py.File(exp_out_dir + '/' + filename, 'r') as f:
                nodes = f['nodes'][:]
                print(np.mean(nodes))
                surface_nodes = f['surface_nodes'][:]
                s.mlab_source.x = nodes[:, 0]
                s.mlab_source.y = nodes[:, 2]
                s.mlab_source.z = nodes[:, 1]
            yield
    anim()
    mlab.show()

def test():
    is_training = False

    o_function = OccFunction('/tmp/pycharm_pcnn_liver/normal_prediction/train_results/2020_07_07_01_26_53/trained_models/model.ckpt')
    pv = o_function.get_data_provider()
    train_generator = pv.provide_data(is_training, 4, 500, 1000)
    for data in train_generator:
        o_function.update_PC(data[0][0])
        while True:
            r = o_function.eval(data[2][0], normal=True)
            print('done')

if __name__ == '__main__':
    if hostname == 'monavale':
        main()
    else:
        plot_result()