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
import scipy
from scipy.sparse.linalg import spsolve, lsmr
import multiprocessing
import time

hostname = socket.gethostname()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from util_package.util import normalize, parse, apply_rigid_transform,\
    angle2rotmatrix, angle2drotmatrix, apply_transform_non_rigid, parse_non_rigid
if hostname != 'monavale':
    from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, \
        plot_meshes, plot_mesh_force, compute_n_plot_force, plot_mesh_points_vector, plot_meshes_n_points
    from mayavi import mlab
else:
    OCC_FUN_DIR = '/tmp/pycharm_pcnn_liver/normal_prediction'
    sys.path.append(OCC_FUN_DIR)
    from Occupancy_Function import OccFunction
from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point

exp_out_dir = '/tmp/exp_out'
num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

class Occ_Nonrigid():
    def __init__(self, points, liver_model, occ_function, deformation_temp, init_para=0):
        self.pc = points
        self.init_para = init_para
        self.point_cloud = points[0]
        self.point_label = points[1]
        self.transformation = points[-1]
        self.GT_mesh = points[-2]
        self.liver_model = liver_model
        #self.liver_model.build_stiffness()
        #self.liver_model.build_laplacian()
        self.occ_function = occ_function
        self.GT_displacement = points[-2]
        self.target_index = points[-3]
        #self.smooth_deformation_templates(deformation_temp)
        self.deformation_templates = deformation_temp
        self.generate_weights()
        self.generate_boundary_vetices_index()

        #self.GT_displacement = points[-2].vertices - self.liver_model.tetr_mesh.vertices

        self.alpha = 0.15

        self.occ_function.update_PC(self.point_cloud, self.point_label)
        #self.reconstructied_mesh = self.occ_function.get_reconstructied_mesh()
        self.smoother = LaplacianSmoothing(self.liver_model.laplacian, 100)
        self.stiffness = liver_model.get_stiffness()
        self.laplacian = liver_model.build_laplacian_volume()
        self.laplacian_solver = 0
        self.step = 0
        # initialization
        #self.transformed_vertices = self.transform(self.liver_model.tetr_mesh.vertices)
        #self.transformed_surface_vertices = self.transform(self.liver_model.surface_mesh.vertices)
        self.transformed_vertices = self.liver_model.tetr_mesh.vertices
        self.transformed_surface_vertices = self.liver_model.surface_mesh.vertices
        self.displacement = np.zeros_like(self.liver_model.tetr_mesh.vertices)
        self.surface_nodes_displacement = np.zeros_like(self.liver_model.surface_mesh.vertices)
        self.initilze_parameters()
        self.least_error = 1.0e100

        subprocess.run(['rm', '-r', exp_out_dir])
        subprocess.run(['mkdir', exp_out_dir])

    def create_solver(self):
        self.solver = pymesh.SparseSolver.create('LDLT')
        self.solver.compute(self.liver_model.stiffness)

    def generate_boundary_vetices_index(self):
        self.surface_boundary_vetices_index = np.where(self.liver_model.surface_nodes_label >= 1)[0]
        self.boundary_vetices_index = self.liver_model.surface_nodes_orgindex[self.surface_boundary_vetices_index]
        unknow_vetices_indicator = np.zeros(shape=self.liver_model.vertices.shape[0])
        unknow_vetices_indicator[self.boundary_vetices_index] = 1
        self.unknow_vetices_index = np.where(unknow_vetices_indicator == 0)[0]

    def generate_weights(self):
        weights = np.ones(shape=self.liver_model.surface_nodes_label.shape, dtype=np.float32)
        #weights[np.where(self.liver_model.surface_nodes_label == 2)] = 1.5
        #weights[np.where(self.liver_model.surface_nodes_label >= 3)] = 0.
        self.weights = self.liver_model.surface_nodes_mass / np.mean(self.liver_model.surface_nodes_mass)
        #self.weights = (weights+self.liver_model.surface_nodes_label) * self.weights

    def smooth_deformation_templates(self, deformation_temp):
        laplacian = self.liver_model.build_laplacian_volume()
        smoother = LaplacianSmoothing(laplacian, 100)

        deformation_templates = np.reshape(deformation_temp, [-1, 42])
        deformation_templates = smoother.filtering(deformation_templates)
        self.deformation_templates = np.reshape(deformation_templates, [-1, 14])

    def save_debug(self, filename):
        return 0
        if not os.path.isfile(exp_out_dir + '/reconstructed_mesh.off'):
            pymesh.save_mesh(exp_out_dir + '/reconstructed_mesh.off', self.reconstructied_mesh)
        with h5py.File(exp_out_dir + '/' + filename, 'w') as f:
            nodes_p = self.apply_registration()
            nodes_p_rigid = self.apply_registration_rigid()
            f.create_dataset('rotation', data=self.pc[-1][0])
            f.create_dataset('scale', data=self.pc[-1][2])
            f.create_dataset('nodes', data=(nodes_p))
            f.create_dataset('nodes_rigid', data=(nodes_p_rigid))
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
        S, T, R, C = parse_non_rigid(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr, deformation = apply_transform_non_rigid(S, T, R, C, node_position_org, self.deformation_templates, self.displacement)
        return node_position

    def apply_registration_rigid(self):
        S, T, R, C = parse_non_rigid(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr, deformation = apply_transform_non_rigid(S, T, R, np.zeros_like(C), node_position_org, self.deformation_templates)
        return node_position

    def evaluate(self):
        error = self.apply_registration() - self.GT_mesh.vertices
        mean_square_error = np.mean(error * error)
        print(mean_square_error)
        #iou = self.compute_IOU()
        #print('IOU: ' + str(iou))

    def evaluate_sparse(self):
        if isinstance(self.GT_displacement, np.ndarray):
            error = self.apply_registration()[self.target_index] - self.GT_displacement
            mean_square_error = np.mean(error * error)
            print(mean_square_error)


    def transform(self, p):
        return np.matmul(self.transformation[0], (p - self.transformation[1]).T).T
    def transform_inv(self, p):
        transformation_inv = np.linalg.inv(self.transformation[0])
        return np.matmul(transformation_inv, p.T).T + self.transformation[1]

    def initilze_parameters(self):
        if self.init_para == 0:
            S, T, R = Iterative_Closest_Point.ICP(self.liver_model, self.pc[0], self.pc[1], self.pc[-1])
            C = np.zeros(shape=14*3)
            self.rigid_parameters = np.concatenate([S, T, R, C], axis=0)
        else:
            S = self.init_para[0]
            T = self.init_para[1]
            R = self.init_para[2]
            C = np.zeros(shape=14*3)
            self.rigid_parameters = np.concatenate([S, T, R, C], axis=0)

    def get_para(self):
        return self.best_para, self.best_displacement

    def rigid_fit(self):
        S, T, R, C = parse_non_rigid(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr = apply_rigid_transform(S, T, R, node_position_org)
        pred, pred_value, pred_normal = self.occ_function.eval(node_position[self.liver_model.surface_nodes_orgindex, :])
        node_position_org = self.transformed_surface_vertices
        #pred_normal = self.smoother.filtering(pred_normal)
        loss1 = np.sum(pred_value * pred_value)
        print('**** ' + str(loss1/pred_value.shape[0]) + '**** ' + str(loss1) + ' **** ')
        if self.least_error > loss1/pred_value.shape[0]:
            self.best_para = self.rigid_parameters
            self.best_displacement = self.displacement
            self.least_error = loss1/pred_value.shape[0]

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

        jacobian = np.concatenate([drdT, drdR3, drdR2, drdR1], axis=1)
        jTj = np.matmul(jacobian.T, jacobian)
        increament = - np.matmul(np.matmul(np.linalg.inv(jTj + 0.0*np.diag(jTj)), jacobian.T), pred_value)
        increament = np.concatenate([np.asarray([0.0]), increament, np.zeros_like(C)], axis=0)
        #increament[0] = 0.0
        self.rigid_parameters += 0.1*increament
        print(self.rigid_parameters)

    def non_rigid_fit_separate(self):
        S, T, R, C = parse_non_rigid(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr, deformation= apply_transform_non_rigid(S, T, R, C, node_position_org, self.deformation_templates, self.displacement)
        pred, pred_value, pred_normal = self.occ_function.eval(node_position[self.liver_model.surface_nodes_orgindex, :])
        node_position_org = self.transformed_surface_vertices + deformation[self.liver_model.surface_nodes_orgindex, :]
        #pred_normal = self.smoother.filtering(pred_normal)
        E = np.sqrt(self.alpha) * np.matmul(np.reshape(deformation, [-1]), self.stiffness.dot(np.reshape(deformation, [-1])))
        loss1 = np.sum(pred_value * pred_value)
        loss2 = E * E
        print('**** ' + str(loss1/pred_value.shape[0]) + '**** ' + str(loss1) + ' **** ' + str(loss2) + ' ****')
        if self.least_error > loss1/pred_value.shape[0]:
            self.best_para = self.rigid_parameters
            self.best_displacement = self.displacement
            self.least_error = loss1/pred_value.shape[0]
        #pred_value = np.tanh(3*pred_value)
        #pred_value = pred_value * self.weights

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

        jacobian1 = np.concatenate([drdT, drdR3, drdR2, drdR1], axis=1)
        #jTj1 = np.matmul(jacobian1.T, jacobian1)
        #increament_rigid = - 1 * np.matmul(np.matmul(np.linalg.inv(jTj1 + 0.00 * np.diag(jTj1)), jacobian1.T), pred_value)
        increament_rigid = -1e-4 * np.mean(jacobian1*np.expand_dims(pred_value, axis=1), axis=0)

        increament = np.concatenate([np.zeros(shape=[1], dtype=np.float64),
                                     increament_rigid,
                                     np.zeros(shape=[14*3], dtype=np.float64)], axis=0)
        #increament = np.concatenate([increament, np.zeros_like(C)], axis=0)
        self.rigid_parameters += 0.1*increament
        print(self.rigid_parameters)

        surface_node_force = -5.0e-6* np.expand_dims(pred_value, axis=1) * pred_normal  # (make sure hav negetive sign)
        self.handle_displacement_laplacian(surface_node_force, S, rr)



    def handle_displacement_laplacian(self, surface_node_force, S, rr):
        surface_node_force = self.smoother.filtering(surface_node_force)
        surface_node_foce_transformed = np.matmul(np.linalg.inv(S * rr), surface_node_force.T).T
        all_node_force = np.zeros_like(self.liver_model.vertices)
        all_node_force[self.liver_model.surface_nodes_orgindex] = surface_node_foce_transformed
        #self.displacement = np.reshape(scipy.sparse.linalg.lsmr(self.stiffness, np.reshape(all_node_movement, [-1])), [-1, 3])

        if isinstance(self.laplacian_solver, int):
            self.create_Laplacian_solver()

        L_bb = self.laplacian[self.unknow_vetices_index, :]
        L_b = L_bb[:, self.boundary_vetices_index]
        rhs = -L_b.dot(all_node_force[self.boundary_vetices_index, :])
        node_force_solution = self.laplacian_solver.solve(rhs)
        displacement_increament = np.zeros_like(self.displacement)
        displacement_increament[self.boundary_vetices_index, :] = all_node_force[self.boundary_vetices_index, :]
        displacement_increament[self.unknow_vetices_index, :] = node_force_solution
        mean_displacement_increament = np.mean(np.sum(displacement_increament*displacement_increament, axis=1))

        if self.step != 0:
            displacement_tmp = np.zeros_like(self.displacement)
            displacement_tmp += self.displacement
            #while mena_norm(displacement_tmp, self.displacement) < 0.5 * mean_displacement_increament:
            for i in range(5):
                stiff_increament = -5e-7 * np.reshape(self.liver_model.stiffness.dot(np.reshape(displacement_tmp, [-1])), [-1,3])
                displacement_tmp += stiff_increament
            self.displacement = displacement_tmp

        self.displacement += displacement_increament
        print(np.mean(np.abs(self.displacement)))

    def create_Laplacian_solver(self):
        L_uu = self.laplacian[self.unknow_vetices_index, :]
        L_u = L_uu[:, self.unknow_vetices_index]
        self.laplacian_solver = pymesh.SparseSolver.create('LDLT')
        self.laplacian_solver.compute(L_u)

    def interpolate_displacement(self):
        boundary_indicator = np.zeros(shape=self.liver_model.vertices.shape[0])
        boundary_indicator[self.boundary_vetices_index] = 1
        order = np.argsort(boundary_indicator)
        num_of_vertices = self.liver_model.vertices.shape[0]
        num_of_boundary = self.boundary_vetices_index.shape[0]

        laplacian_i = self.laplacian[order[:(num_of_vertices - num_of_boundary)], :]
        laplacian_ii = laplacian_i[:, order[:(num_of_vertices - num_of_boundary)]]
        laplacian_ib = laplacian_i[:, order[(num_of_vertices - num_of_boundary):]]
        # plot_mesh_points(mesh.vertices, mesh.faces, vertices_sorted[(num_of_vertices-num_of_boundary):, :])
        free_displacement1 = scipy.sparse.linalg.spsolve(laplacian_ii, -laplacian_ib.dot(self.displacement[order[(num_of_vertices - num_of_boundary):], :]))

        #free_displacement2 = scipy.sparse.linalg.lsqr(laplacian_ii, -laplacian_ib.dot(self.displacement[order[(num_of_vertices - num_of_boundary):], 0]))
        self.displacement[order[:(num_of_vertices - num_of_boundary)], :] = free_displacement1
        return


    def evaluate_points2surface(self):
        S, T, R, C = parse_non_rigid(self.rigid_parameters)
        node_position_org = self.transformed_vertices
        node_position, rr, deformation = apply_transform_non_rigid(S, T, R, C, node_position_org,
                                                                   self.deformation_templates)
        surface_vertices = node_position[self.liver_model.surface_nodes_orgindex, :]
        mesh = trimesh.Trimesh(vertices=surface_vertices, faces=self.liver_model.surface_mesh.faces)
        closest_p, distance, face_id = trimesh.proximity.closest_point_naive(mesh, self.point_cloud)
        print(np.mean(distance))

    def run(self, num_iter, save_result=False):
        for i in range(num_iter):
            print('********* step: '+str(i) + ' **********')
            if i % 1 == 0:
                self.save_debug(str(i).zfill(5)+'.h5')
            #self.evaluate_points2surface()
            if i < 100:
                self.non_rigid_fit_separate()
            else:
                #self.rigid_fit()
                self.non_rigid_fit_separate()
            '''nodes_p = self.apply_registration()
            surface_nodes_p = nodes_p[self.liver_model.surface_nodes_orgindex, :]

            pred, pred_value, pred_normal = self.occ_function.eval(surface_nodes_p)
            #surface_node_force = -1.0e-4* np.expand_dims(np.tanh(3*pred_value), axis=1) * pred_normal
            surface_node_force = -3.0e-5 * np.expand_dims(pred_value, axis=1) * pred_normal #(make sure hav negetive sign)
            if i < 20:
                surface_node_force_smoothed = np.zeros_like(surface_node_force)
            else:
                surface_node_force_smoothed = self.smoother.filtering(surface_node_force)
            #surface_node_force_smoothed = np.zeros_like(surface_node_force)
            node_force = np.zeros_like(self.displacement)
            node_force[self.liver_model.surface_nodes_orgindex, :] = surface_node_force_smoothed

            self.displacement += node_force
            self.surface_nodes_displacement = self.displacement[self.liver_model.surface_nodes_orgindex, :]'''
            self.step += 1

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

def mena_norm(input, in_org):
    displacement_diff = input - in_org
    mean = np.mean(np.sum(displacement_diff * displacement_diff, axis=1))
    return mean


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

    from util_package.plot import plot_points_label, plot_mesh_points
    plot_points_label(pc[0][:pc[-3], :], pc[1][:pc[-3]])

    filename = str(499).zfill(5) + '.h5'
    reconstructed_mesh = pymesh.load_mesh(exp_out_dir + '/reconstructed_mesh.off')
    with h5py.File(exp_out_dir + '/' + filename, 'r') as f:
        nodes = f['nodes'][:]
        surface_nodes = f['surface_nodes'][:]
        s = plot_meshes_n_points(reconstructed_mesh.vertices, reconstructed_mesh.faces,#np.matmul(pc[-1][0], (pc[-2].vertices - pc[-1][1]).T).T, pc[-2].faces,
                                 nodes, liver_model.tetr_mesh.faces,
                                 pc[0][:pc[-3], :], show=True)
    filename = str(0).zfill(5) + '.h5'
    with h5py.File(exp_out_dir + '/' + filename, 'r') as f:
        nodes = f['nodes'][:]
        surface_nodes = f['surface_nodes'][:]
        s = plot_meshes_n_points(reconstructed_mesh.vertices, reconstructed_mesh.faces,#np.matmul(pc[-1][0], (pc[-2].vertices - pc[-1][1]).T).T, pc[-2].faces,
                                 nodes, liver_model.tetr_mesh.faces,
                                 pc[0][:pc[-3], :], show=True)

    @mlab.animate
    def anim():
        for i in range(1, 100):
            if i % 1 != 0:
                continue
            filename = str(i).zfill(5) + '.h5'
            with h5py.File(exp_out_dir + '/' + filename, 'r') as f:
                nodes = f['nodes'][:]
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