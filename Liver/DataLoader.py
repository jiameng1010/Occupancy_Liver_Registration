import os
import sys
import numpy as np
import h5py
import trimesh
import pymesh
#import pymesh
from trimesh import sample, ray, triangles
from trimesh.ray.ray_triangle import RayMeshIntersector, ray_triangle_id
BASE_DIR = './'
sys.path.append(BASE_DIR)
from random import shuffle
import socket
import string
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=5)
#from mayavi import mlab

#from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, \
#        plot_meshes, plot_mesh_force, compute_n_plot_force, plot_mesh_points_vector

def handle_punctuation(input):
    output = input.replace('.', '我')
    output = output.translate(str.maketrans('', '', string.punctuation))
    output = output.replace('我', '.')
    output = output.replace(' ', '_')
    return output


class ProbeProvider():
    def __init__(self, cfg):
        data_dir = cfg['basics']['dataset_root']
        self.training_data_dir = data_dir + '/train' + handle_punctuation(\
            str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para']))
        self.validation_data_dir = data_dir + '/val' + handle_punctuation(\
            str(cfg['basics']['total_training'])+str(cfg['deformation_randomness'])+str(cfg['sample_para']))
        self.num_train_files = cfg['basics']['total_training']
        self.num_val_files = cfg['basics']['total_validation']
        self.cfg = cfg['training']

    def getValDataFiles(self):
        return self.getDataFiles(self.val_files)

    def getTrainDataFiles(self):
        return self.getDataFiles(self.train_files)

    def getDataFiles(self,list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def load_h5(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def loadDataFile(self,filename):
        return self.load_h5(os.path.join(os.path.join(BASE_DIR,'hdf5_data')),filename)
        #verts,_ = read_off(filename)
        #return verts

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        return verts, faces

    def load_h5_data_label_seg(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:].astype(np.int32)
        seg = f['pid'][:]
        return (data, label, seg)

    def translate_point_cloud3(self,batch_data1, batch_data2, batch_data3,):

        translated_data1 = np.zeros(batch_data1.shape, dtype=np.float32)
        translated_data2 = np.zeros(batch_data2.shape, dtype=np.float32)
        translated_data3 = np.zeros(batch_data3.shape, dtype=np.float32)
        for k in range(batch_data1.shape[0]):
            xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
            xyz2 = np.random.uniform(low=-0.2,high=0.2,size=[3])

            shape_pc1 = batch_data1[k, ...]
            shape_pc2 = batch_data2[k, ...]
            shape_pc3 = batch_data3[k, ...]

            translated_data1[k, ...] = np.add(np.multiply(shape_pc1, xyz1), xyz2)
            translated_data2[k, ...] = np.add(np.multiply(shape_pc2, xyz1), xyz2)
            translated_data3[k, ...] = np.add(np.multiply(shape_pc3, xyz1), xyz2)
        return translated_data1, translated_data2, translated_data3


    def loadDataFile_with_seg(self,filename):
        return self.load_h5_data_label_seg(filename)

    def load_one_pointcloud(self, is_train, id):
        if is_train:
            dir = self.training_data_dir
        else:
            dir = self.validation_data_dir
        h5f = h5py.File(dir + '/' + str(id).zfill(6) + "/points.h5", 'r')
        points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout = load_data(h5f)
        h5f.close()

        #
        from util_package.plot import plot_pointses, plot_pointss
        points_to_plot = np.concatenate([points_around, points_uniform], axis=0)
        points_to_plot_label = np.concatenate([around_inout, uniform_inout], axis=0)
        plot_pointss(points_to_plot, points_to_plot)
        plot_pointses(points_to_plot[np.where(points_to_plot_label == False)[0], :], points_to_plot[np.where(points_to_plot_label == True)[0], :])
        #

        num_on = self.cfg['num_points1'] + int(np.random.uniform() * self.cfg['num_points2'])
        points_on_front, points_on_back, points_on_FF, points_on_LR, points_on_RR = parse_points(points_on,
                                                                                                 points_part_label)
        points_candidates = np.concatenate([points_on_FF, points_on_RR, points_on_LR], axis=0)
        label_candidates = np.concatenate([2*np.ones(shape=points_on_FF.shape[0], dtype=np.int16),
                                           4*np.ones(shape=points_on_RR.shape[0], dtype=np.int16),
                                           3*np.ones(shape=points_on_LR.shape[0], dtype=np.int16)], axis=0)
        p = np.random.permutation(points_candidates.shape[0])
        points_candidates = points_candidates[p, :]
        output_on = points_candidates[:num_on, :]
        output_label = label_candidates[p[:num_on]]
        num_around = int(self.cfg['around_uniform_ratio'] * self.cfg['num_points2'])
        output_around = points_around[:num_around, :]
        output_inout_1 = around_inout[:num_around]
        output_uniform = points_uniform[:((self.cfg['num_sample_points'] - num_around)), :]
        output_inout_2 = uniform_inout[:((self.cfg['num_sample_points'] - num_around))]
        output_all = np.concatenate([output_around, output_uniform], axis=0)
        output_inout = np.concatenate([output_inout_1, output_inout_2], axis=0)

        output_on, output_all, transformations = scale_rotation(output_on, output_all, self.cfg['scale_to'], True)
        deformed_mesh = pymesh.load_mesh(dir + '/' + str(id).zfill(6) + "/mesh.off")
        #plot_mesh_points(np.matmul(transformations[0], (deformed_mesh.vertices - transformations[1]).T).T,
        #                 deformed_mesh.faces, output_on)
        output_on = np.concatenate([output_on, (5 * self.cfg['scale_to']) * np.ones(
            shape=(self.cfg['num_points1'] + self.cfg['num_points2'] - num_on, 3), dtype=np.float32)], axis=0)
        output_label = np.concatenate([output_label,
                                       np.zeros(shape=(self.cfg['num_points1'] + self.cfg['num_points2'] - num_on),
                                                dtype=np.int32)], axis=0)
        return output_on, output_label, output_all, output_inout, num_on, deformed_mesh, transformations

    def load_one_pointcloud_real(self, is_train, id):
        if is_train:
            dir = self.training_data_dir
        else:
            dir = self.validation_data_dir
        h5f = h5py.File(dir + '/' + str(id).zfill(6) + "/points.h5", 'r')
        points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout = load_data(h5f)
        h5f.close()

        #
        from util_package.plot import plot_pointses, plot_pointss
        points_to_plot = np.concatenate([points_around, points_uniform], axis=0)
        points_to_plot_label = np.concatenate([around_inout, uniform_inout], axis=0)
        plot_pointss(points_to_plot, points_to_plot)
        plot_pointses(points_to_plot[np.where(points_to_plot_label == False)[0], :], points_to_plot[np.where(points_to_plot_label == True)[0], :])
        #

        num_on = self.cfg['num_points1'] + int(np.random.uniform() * self.cfg['num_points2'])
        points_on_front, points_on_back, points_on_FF, points_on_LR, points_on_RR = parse_points(points_on,
                                                                                                 points_part_label)
        points_candidates = np.concatenate([points_on_FF, points_on_RR, points_on_LR], axis=0)
        label_candidates = np.concatenate([2*np.ones(shape=points_on_FF.shape[0], dtype=np.int16),
                                           4*np.ones(shape=points_on_RR.shape[0], dtype=np.int16),
                                           3*np.ones(shape=points_on_LR.shape[0], dtype=np.int16)], axis=0)
        p = np.random.permutation(points_candidates.shape[0])
        points_candidates = points_candidates[p, :]
        output_on = points_candidates[:num_on, :]
        output_label = label_candidates[p[:num_on]]
        num_around = int(self.cfg['around_uniform_ratio'] * self.cfg['num_points2'])
        output_around = points_around[:num_around, :]
        output_inout_1 = around_inout[:num_around]
        output_uniform = points_uniform[:((self.cfg['num_sample_points'] - num_around)), :]
        output_inout_2 = uniform_inout[:((self.cfg['num_sample_points'] - num_around))]
        output_all = np.concatenate([output_around, output_uniform], axis=0)
        output_inout = np.concatenate([output_inout_1, output_inout_2], axis=0)

        output_on, output_all, transformations = scale_rotation(output_on, output_all, self.cfg['scale_to'], True)
        deformed_mesh = pymesh.load_mesh(dir + '/' + str(id).zfill(6) + "/mesh.off")
        #plot_mesh_points(np.matmul(transformations[0], (deformed_mesh.vertices - transformations[1]).T).T,
        #                 deformed_mesh.faces, output_on)
        output_on = np.concatenate([output_on, (5 * self.cfg['scale_to']) * np.ones(
            shape=(self.cfg['num_points1'] + self.cfg['num_points2'] - num_on, 3), dtype=np.float32)], axis=0)
        output_label = np.concatenate([output_label,
                                       np.zeros(shape=(self.cfg['num_points1'] + self.cfg['num_points2'] - num_on),
                                                dtype=np.int32)], axis=0)
        return output_on, output_label, output_all, output_inout, num_on, deformed_mesh, transformations

    def provide_data(self, is_train, BATCH_SIZE, NUM_POINT, NUM_PROBE, shuffle=True):
        if is_train:
            dir = self.training_data_dir
            num = self.num_train_files
            if shuffle:
                indices = np.random.permutation(num)
            else:
                indices = np.arange(num)
        else:
            dir = self.validation_data_dir
            num = self.num_val_files
            if shuffle:
                indices = np.random.permutation(num)
            else:
                indices = np.arange(num)

        output_pointcloud = np.zeros(shape=(self.cfg['batch_size'], (self.cfg['num_points1']+self.cfg['num_points2']), 3), dtype=np.float32)
        output_point_label = np.ones(shape=(self.cfg['batch_size'], (self.cfg['num_points1']+self.cfg['num_points2'])), dtype=np.int32)
        output_probepoint = np.zeros(shape=(self.cfg['batch_size'], self.cfg['num_sample_points'], 3), dtype=np.float32)
        output_inout_label = np.ones(shape=(self.cfg['batch_size'], self.cfg['num_sample_points']), dtype=np.int32)
        filled = 0
        for i in range(num):
            try:
                ii = indices[i]
                h5f = h5py.File(dir+'/'+str(ii).zfill(6) + "/points.h5", 'r')
            except:
                print('training file not exist')
                continue
            else:
                points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout = load_data(h5f)
                h5f.close()

                num_on = self.cfg['num_points1'] + int(np.random.uniform() * self.cfg['num_points2'])
                points_on_front, points_on_back, points_on_FF, points_on_LR, points_on_RR = parse_points(points_on, points_part_label)
                points_candidates = np.concatenate([points_on_FF, points_on_RR, points_on_LR], axis=0)
                p = np.random.permutation(points_candidates.shape[0])
                points_candidates = points_candidates[p, :]
                output_on = points_candidates[:num_on, :]
                output_label = points_part_label[:num_on]
                num_around = int(self.cfg['around_uniform_ratio'] * self.cfg['num_points2'])
                output_around = points_around[:num_around, :]
                output_inout_1 = around_inout[:num_around]
                output_uniform = points_uniform[:((self.cfg['num_sample_points']-num_around)), :]
                output_inout_2 = uniform_inout[:((self.cfg['num_sample_points']-num_around))]
                output_all = np.concatenate([output_around, output_uniform], axis=0)
                output_inout = np.concatenate([output_inout_1, output_inout_2], axis=0)

                output_on, output_all = scale_rotation(output_on, output_all, self.cfg['scale_to'])
                output_on = np.concatenate([output_on, (5*self.cfg['scale_to'])*np.ones(shape=(self.cfg['num_points1']+self.cfg['num_points2']-num_on, 3), dtype=np.float32)], axis=0)
                output_label = np.concatenate([output_label, np.zeros(shape=(self.cfg['num_points1']+self.cfg['num_points2']-num_on), dtype=np.int32)], axis=0)

                output_pointcloud[filled, :, :] = output_on
                output_point_label[filled, :] = output_label
                output_probepoint[filled, :, :] = output_all
                output_inout_label[filled, :] = output_inout

                filled += 1
                h5f.close()
                if filled == BATCH_SIZE:
                    filled = 0
                    yield output_pointcloud, output_point_label, output_probepoint, output_inout_label

def parse_points(points_on, points_part_label):
    index_front = np.where(points_part_label != 0)
    points_on_front = points_on[index_front[0], :]
    index_back = np.where(points_part_label == 0)
    points_on_back = points_on[index_back[0], :]
    index_FF = np.where(points_part_label == 2)
    points_on_FF = points_on[index_FF[0], :]
    index_LR = np.where(points_part_label == 3)
    points_on_LR = points_on[index_LR[0], :]
    index_RR = np.where(points_part_label == 4)
    points_on_RR = points_on[index_RR[0], :]
    return points_on_front, points_on_back, points_on_FF, points_on_LR, points_on_RR


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

def scale_rotation(input_on, input_all, size, provide_transformation=False):
    # shift and scale
    max = np.max(input_on, axis=0)
    min = np.min(input_on, axis=0)
    center = (max + min)/2
    # rotation
    r1, r2, r3, rotation_matrix = angle2rotmatrix(np.random.uniform(0, 2*np.pi, size=3))
    output_on = np.matmul(rotation_matrix, (input_on-center).T).T
    output_all = np.matmul(rotation_matrix, (input_all-center).T).T

    max = np.max(output_on, axis=0)
    min = np.min(output_on, axis=0)
    center2 = (max + min)/2
    scale = size / np.max(max - center2)
    output_on = scale * (output_on)
    output_all = scale * (output_all)

    if provide_transformation:
        return output_on, output_all, ((scale)*rotation_matrix, center)
    else:
        return output_on, output_all


def load_data(h5f):
    points_on = h5f['points_on'][:]
    p1 = np.random.permutation(points_on.shape[0])
    points_on = points_on[p1,:]
    points_part_label = h5f['points_part_label'][:]
    points_part_label = points_part_label[p1]

    points_around = h5f['points_around'][:]
    p2 = np.random.permutation(points_around.shape[0])
    points_around = points_around[p2, :]
    around_inout = h5f['around_inout'][:]
    around_inout = around_inout[p2]

    points_uniform = h5f['points_uniform'][:]
    p3 = np.random.permutation(points_around.shape[0])
    points_uniform = points_uniform[p3, :]
    uniform_inout = h5f['uniform_inout'][:]
    uniform_inout = uniform_inout[p3]
    return points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout

def load_processed_shapenet(pointcloud_f, points_f, NUM_POINT, NUM_PROBE):
    pointcloud_data = np.load(pointcloud_f)
    points_data = np.load(points_f)

    if pointcloud_data['scale'] == points_data['scale']:
        scale = pointcloud_data['scale']
    else:
        raise Exception('scale not equal')

    if any(pointcloud_data['loc'] != np.asarray([0,0,0])) or any(points_data['loc'] != np.asarray([0,0,0])):
        raise Exception('loc not zero')

    pointcloud = pointcloud_data['points']
    probe_points = points_data['points']
    np.random.shuffle(pointcloud)
    label = np.unpackbits(points_data['occupancies'], axis=0)
    index_in = np.where(label == 1)[0]
    index_out = np.where(label ==0)[0]
    if len(index_in) < NUM_PROBE or len(index_out) < NUM_PROBE:
        raise Exception('too less points')
    np.random.shuffle(index_in)
    np.random.shuffle(index_out)

    output_pointcloud = pointcloud[:NUM_POINT, :] + np.random.normal(scale=0.05, size=(NUM_POINT, 3))# / scale
    output_points1 = probe_points[index_in[:NUM_PROBE], :]# /scale
    output_points2 = probe_points[index_out[:NUM_PROBE], :]
    output_points = np.concatenate([output_points1, output_points2], axis=0)
    output_label = np.concatenate([np.ones(shape=NUM_PROBE), np.zeros(shape=NUM_PROBE)], axis=0)
    #plot(output_pointcloud)
    #plot(output_points[np.where(output_label==1)[0], :])
    return 2*output_pointcloud, 2*output_points, output_label

def random_view_point():
    direction = np.empty((3))
    direction[2] = 0.5 * np.random.rand()
    direction[1] = np.random.rand() - 0.5
    direction[0] = np.random.rand() - 0.5
    rangm = 10 * (1 + np.random.rand())

    # random view point
    viewpoint = rangm * direction / np.linalg.norm(direction)
    return viewpoint


def view_partial_sample(id, n):
    viewpoint = random_view_point()
    # this method retures a n*3 array that indicates the pionts sampled from the visible surface
    mesh = trimesh.load(id)
    location_v = np.empty((n, 3))
    number_of_visibles = 0
    r = RayMeshIntersector(mesh)


    init_sample = sample.sample_surface(mesh, int(n/4))[0]

    ray_origins = np.tile(np.expand_dims(viewpoint, axis=0), [n, 1])
    ray_directions = np.concatenate([init_sample]*4, axis=0) - ray_origins

    [index_tri, index_ray, location] = r.intersects_id(ray_origins, ray_directions, return_locations=True,
                                                       multiple_hits=False)
    scale = np.max(mesh.bounding_box.vertices[:, :])

    return (0.5 / scale)*location
    # delete unseen
    # visible_index = []
    '''for i in range(index_ray.shape[0]):
        if not (abs(location[i, 0] - init_sample[index_ray[i], 0]) + abs(
                location[i, 1] - init_sample[index_ray[i], 1]) + abs(
            location[i, 2] - init_sample[index_ray[i], 2])) > 0.0001:
            location_v[number_of_visibles, :] = location[i, :]
            number_of_visibles += 1
            if number_of_visibles == n:
                return location_v'''

def partial_sample(points_on):
    direction = np.empty((3))
    direction[2] = 0.5 * np.random.rand()
    direction[1] = np.random.rand() - 0.5
    direction[0] = np.random.rand() - 0.5
    norm = np.linalg.norm(direction)
    direction = norm * direction
    rotated_points = np.matmul(points_on, direction)
    points_out = np.squeeze(points_on[np.argwhere(rotated_points > 0), :])

    return points_out


#load_processed_shapenet('/home/mjia/Documents/jointSegmentation/Occupancy-Networks/data_org/ShapeNet/02691156/d2412f19d33572dc4c3a35cee92bb95b/pointcloud.npz',\
#                        '/home/mjia/Documents/jointSegmentation/Occupancy-Networks/data_org/ShapeNet/02691156/d2412f19d33572dc4c3a35cee92bb95b/points.npz', 300, 400)

#pv = ProbeProvider()
#train_generator = pv.provide_processed_shapenet_data(True, 5, 300, 400)
#for a in train_generator:
#    print(' ')
