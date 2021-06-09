from mayavi import mlab
import numpy as np

#points: n*3
def plot_points(points):
    mlab.points3d(points[:,0], points[:,1], points[:,2], scale_factor=1)
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
#points: n*3
def plot_mesh_points(mesh_v, mesh_f, points):
    scale = 1
    f = mlab.figure()
    mlab.points3d(points[:,0], points[:,2], points[:,1], scale_factor=0.01)
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

def plot_mesh_points_label(mesh_v, meshes, points, PC_label):
    scale = 1
    f = mlab.figure()
    points_FF = points[np.where(PC_label==0), :][0]
    points_LR = points[np.where(PC_label==1), :][0]
    points_RR = points[np.where(PC_label==2), :][0]
    points_SR = points[np.where(PC_label==3), :][0]
    mlab.points3d(points_FF[:,0], points_FF[:,2], points_FF[:,1], scale_factor=0.01, color=(1, 0, 0))
    mlab.points3d(points_LR[:,0], points_LR[:,2], points_LR[:,1], scale_factor=0.01, color=(0, 1, 0))
    mlab.points3d(points_RR[:,0], points_RR[:,2], points_RR[:,1], scale_factor=0.01, color=(0, 0, 1))
    mlab.points3d(points_SR[:,0], points_SR[:,2], points_SR[:,1], scale_factor=0.01, color=(0, 1, 1))
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[0].faces,
                         opacity=0.5,
                         color=(1, 0, 0))#FF
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[1].faces,
                         opacity=0.5,
                         color=(0, 1, 0))
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[2].faces,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[3].faces,
                         opacity=0.5,
                         color=(0, 1, 1))
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
def plot_mesh(mesh_v, mesh_f):
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
#mean: 1*3
#normal: 3*3
def plot_mesh_surfacenormal(mesh_v, mesh_f, mean, normal):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    origin = np.tile(mean, [3,1])
    mlab.quiver3d(origin[0, 0], origin[0, 1], origin[0, 2], normal[0,0], normal[1,0], normal[2,0],
                  scale_factor=0.2, color = (1,0,0))
    mlab.quiver3d(origin[0, 0], origin[0, 1], origin[0, 2], normal[0, 1], normal[1, 1], normal[2, 1],
                  scale_factor=0.2, color = (0,1,0))
    mlab.quiver3d(origin[0, 0], origin[0, 1], origin[0, 2], normal[0, 2], normal[1, 2], normal[2, 2],
                  scale_factor=0.2, color = (0,0,1))

    mlab.show()

def plot_displacement(mesh_back, displacement_boundary):
    mlab.triangular_mesh([vert[0] for vert in mesh_back.vertices],
                         [vert[1] for vert in mesh_back.vertices],
                         [vert[2] for vert in mesh_back.vertices],
                         mesh_back.faces,
                         opacity=0.5,
                         color=(0, 0, 1))
    vertices_deformed = mesh_back.vertices + displacement_boundary
    mlab.triangular_mesh([vert[0] for vert in vertices_deformed],
                         [vert[1] for vert in vertices_deformed],
                         [vert[2] for vert in vertices_deformed],
                         mesh_back.faces,
                         opacity=0.5,
                         color=(1, 0, 1))
    mlab.show()
