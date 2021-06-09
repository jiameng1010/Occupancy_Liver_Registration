import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
import trimesh
from trimesh import sample

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def closest(A, B):
    distance = np.expand_dims(A, axis=0) - np.expand_dims(B, axis=1)
    distance = np.sum(distance * distance, axis=2)
    closest = np.argmin(distance, axis=0)
    return distance[closest, np.arange(A.shape[0])], closest


def find_closest(A, A_label, B, B_label):
    A_FF = A[np.where(A_label==2)[0], :]
    B_FF = B[np.where(B_label==2)[0], :]
    d_FF, c_FF = closest(A_FF, B_FF)

    A_LR = A[np.where(A_label==3)[0], :]
    B_LR = B[np.where(B_label==3)[0], :]
    d_LR, c_LR = closest(A_LR, B_LR)

    A_RR = A[np.where(A_label==4)[0], :]
    B_RR = B[np.where(B_label==4)[0], :]
    d_RR, c_RR = closest(A_RR, B_RR)

    distance = np.zeros(shape=A.shape[0])
    distance[np.where(A_label == 2)[0]] = d_FF
    distance[np.where(A_label == 3)[0]] = d_LR
    distance[np.where(A_label == 4)[0]] = d_RR
    index = np. zeros(shape=A.shape[0], dtype=np.int16)
    index[np.where(A_label == 2)[0]] = np.where(B_label==2)[0][c_FF]
    index[np.where(A_label == 3)[0]] = np.where(B_label==3)[0][c_LR]
    index[np.where(A_label == 4)[0]] = np.where(B_label==4)[0][c_RR]
    return distance, index


def icp(A, A_label, B, B_label, init_pose=None, max_iterations=200, tolerance=0.0000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    #assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        #distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        distances, indices = find_closest(src[:m, :].T, A_label, dst[:m, :].T, B_label)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        #if np.abs(prev_error - mean_error) < tolerance:
        #    break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    import h5py
    with h5py.File('debug_points.h5', 'w') as f:
        f.create_dataset('src', data=src.T[:, :3])
        f.create_dataset('dst', data=dst.T[:, :3])
    return T, distances, i

def retore_angles(R):
    pitch = np.arcsin(-R[2, 0])
    roll = np.arcsin(R[2, 1] / np.cos(pitch))
    yaw = np.arcsin(R[1, 0] / np.cos(pitch))

    #debug
    #from util_package.util import angle2rotmatrix
    #debug_a = R - angle2rotmatrix(np.asarray([roll, pitch, yaw]))[3]
    return np.asarray([roll, pitch, yaw])

def ICP(liver_model, pc, pc_label, pc_trans, S=1):
    #S = 1#np.random.normal(loc=1.0, scale=0.1, size=20)
    pointcloud = pc[np.where(pc_label >= 2), :][0]
    pointcloud_label = pc_label[np.where(pc_label >= 2)[0]]

    #transformed_vertices = np.matmul(pc1[-1][0], (liver_model.tetr_mesh.vertices - pc1[-1][1]).T).T
    scale, shift = liver_model.set_scale(pc_trans[-1])
    transformed_vertices = liver_model.shift_n_scale(liver_model.tetr_mesh.vertices)
    faces = liver_model.tetr_mesh.faces[np.where(liver_model.face_label >= 2), :][0]
    face_label = liver_model.face_label[np.where(liver_model.face_label >= 2)[0]]
    mesh = trimesh.Trimesh(transformed_vertices, faces)
    points, face_index = sample.sample_surface(mesh, 2 * pointcloud.shape[0])
    points_label = face_label[face_index]
    T, d, _ = icp(points, points_label, pointcloud, pointcloud_label)

    R = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    R2 = retore_angles(T[:3, :3])

    #from util_package.plot import plot_mesh_points, plot_points_hightlight
    from util_package.util import angle2rotmatrix, apply_rigid_transform
    import h5py
    src = np.ones((4,transformed_vertices.shape[0]))
    src[:3,:] = np.copy(transformed_vertices.T)
    vertices1 = np.dot(T, src)

    vertices2, _ = apply_rigid_transform(np.asarray([scale]),  T[:3, 3]-scale*np.matmul(T[:3, :3], shift), R,
                                         liver_model.tetr_mesh.vertices)
    #plot_mesh_points(vertices2, liver_model.tetr_mesh.faces, pc)
    #with h5py.File('debug.h5', 'w') as f:
    #    f.create_dataset('vertices2', data=vertices2)
    #    f.create_dataset('faces', data=liver_model.tetr_mesh.faces)
    #    f.create_dataset('pc', data=pointcloud)
    #    f.create_dataset('points', data=points)
    return np.asarray([scale]),  T[:3, 3]-scale*np.matmul(T[:3, :3], shift), R





def test():
    import yaml
    from LiverModel import LiverModel
    from DataLoader import ProbeProvider
    from util_package.plot import plot_meshes_n_points, plot_points_hightlight
    from util_package.util import angle2rotmatrix, apply_rigid_transform
    from sparse_dataset.Sparse_Data import Sparse_Data_loader

    with open('debug.yaml', 'r') as f:
        cfg = yaml.load(f)
    data_loader = Sparse_Data_loader(cfg)
    output_points, output_label, num_on, GT_target, transformations = data_loader.load(4)
    pointcloud = output_points[np.where(output_label >= 2), :][0]
    R = angle2rotmatrix(np.random.normal(size=[3]))
    pointcloud1 = np.matmul(R[-1], pointcloud.T).T
    pointcloud_label = output_label[np.where(output_label >= 2)[0]]

    liver_model = LiverModel()
    scale, shift = liver_model.set_scale(np.max(pointcloud))
    transformed_vertices = liver_model.shift_n_scale(liver_model.tetr_mesh.vertices)
    faces = liver_model.tetr_mesh.faces[np.where(liver_model.face_label >= 2), :][0]
    face_label = liver_model.face_label[np.where(liver_model.face_label >= 2)[0]]
    mesh = trimesh.Trimesh(transformed_vertices, faces)
    plot_meshes_n_points(mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, pointcloud1)
    points, face_index = sample.sample_surface(mesh, 2*pointcloud1.shape[0])
    points_label = face_label[face_index]

    #plot_points_hightlight(points, points[np.where(points_label==3)[0], :])
    S = np.random.normal(loc=0.2, scale=0.01, size=20)
    S = 1
    T, d, _ = icp(S*points, points_label, pointcloud1, pointcloud_label)
    R = retore_angles(T[:3, :3])
    R = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    #R = np.pi * R/180
    R_matrix = angle2rotmatrix(R)[-1]

    debug = T[:3, :3] - angle2rotmatrix(Rotation.from_matrix(T[:3, :3]).as_euler('xyz'))[-1]
    #S = 1
    #T = T[:3, 3]

    src = np.ones((4, transformed_vertices.shape[0]))
    src[:3, :] = np.copy(transformed_vertices.T)
    vertices = np.dot(T, src).T[:, :3]
    vertices1, _ = apply_rigid_transform(S, T[:3, 3], R, transformed_vertices)

    vertices2, _ = apply_rigid_transform(np.asarray([scale]),  T[:3, 3]-scale*np.matmul(T[:3, :3], shift), R, liver_model.tetr_mesh.vertices)
    mesh = trimesh.Trimesh(vertices, faces)
    plot_meshes_n_points(mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, pointcloud1)

def plot_debug():
    from LiverModel import LiverModel
    liver_model = LiverModel()
    import h5py
    with h5py.File('debug.h5', 'r') as f:
        vertices2 = f['vertices2'][:]
        #vertices2 = vertices2.T[:, :3]
        faces = f['faces'][:]
        pc = f['pc'][:]

    with h5py.File('debug_points.h5', 'r') as f:
        pc1 = f['src'][:]
        pc2 = f['dst'][:]

    from util_package.plot import plot_mesh_points, plot_points_hightlight, plot_pointss
    plot_pointss(pc1, pc2)
    plot_mesh_points(vertices2, faces, pc)


if __name__ == '__main__':
    plot_debug()
    print('done')