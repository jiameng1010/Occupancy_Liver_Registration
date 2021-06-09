import numpy as np
import trimesh
from trimesh import sample

class Sampler():
    def __init__(self, cfg):
        self.scale = cfg['Gaussian_scale']
        self.num_onsurface_points = cfg['num_onsurface_points']
        self.num_around_surface = cfg['num_around_surface']
        self.num_uniform = cfg['num_uniform']
        self.bbox_padding = cfg['bbox_padding']

    def sample(self, mesh_in, label):
        mesh = trimesh.Trimesh(mesh_in.vertices, mesh_in.faces)
        points_on, faces_id = sample.sample_surface(mesh, self.num_onsurface_points)
        points_part_label = label[faces_id]

        points_around, faces_id = sample.sample_surface(mesh, self.num_around_surface)
        points_around = points_around + np.random.normal(scale=self.scale, size=points_around.shape)
        around_inout = mesh.contains(points_around)

        bbox = mesh.vertices[:,0]
        range = max(bbox) - min(bbox)
        points_uniform0 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,1]
        range = max(bbox) - min(bbox)
        points_uniform1 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        bbox = mesh.vertices[:,2]
        range = max(bbox) - min(bbox)
        points_uniform2 = np.random.uniform(min(bbox)-0.15*range,
                                            max(bbox)+0.15*range,
                                            self.num_uniform)
        points_uniform = np.concatenate([np.expand_dims(points_uniform0, axis=1),
                                         np.expand_dims(points_uniform1, axis=1),
                                         np.expand_dims(points_uniform2, axis=1)],
                                         axis=1)
        uniform_inout = mesh.contains(points_uniform)
        return points_on, points_part_label, points_around, around_inout, points_uniform, uniform_inout
