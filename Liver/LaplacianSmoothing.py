import scipy
import numpy as np
if __name__ == '__main__':
    from LiverModel import LiverModel
    from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label, \
        plot_meshes, plot_mesh_force, compute_n_plot_force, plot_mesh_points_vector, plot_meshes_n_points,\
        plot_mesh_function

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class LaplacianSmoothing():
    def __init__(self, L, n):
        '''eigen_values, eigen_vectors = scipy.linalg.eigh(L.toarray())
        index = np.argsort(eigen_values)[::-1]
        self.eigen_values = eigen_values[index[:n]]
        self.eigen_vectors = eigen_vectors[:, index[:n]]'''

        eigen_values, eigen_vectors = scipy.sparse.linalg.eigsh(L, n+1, which='SA')
        self.eigen_values = eigen_values[1:]
        self.eigen_vectors = eigen_vectors[:, 1:]
        x = np.linspace(0, 1, n)
        self.filter_coefficients = gaussian(x, 0.0, 0.3)

    def filtering(self, signal):
        coeffecients = self.eigen_vectors.T.dot(signal)
        coeffecients = np.expand_dims(self.filter_coefficients, axis=1) * coeffecients
        return self.eigen_vectors.dot(coeffecients)


def main():
    liver_model = LiverModel()
    liver_model.build_laplacian()
    liver_model.surface_mesh.add_attribute("vertex_normal")
    surface_function = liver_model.surface_mesh.get_attribute('vertex_normal')
    surface_function = np.reshape(surface_function, [-1, 3])
    plot_mesh_function(liver_model.surface_mesh.vertices, liver_model.surface_mesh.faces, surface_function[:,0])

    smoother = LaplacianSmoothing(liver_model.laplacian, 50)
    function_smoothed = smoother.filtering(surface_function)
    plot_mesh_function(liver_model.surface_mesh.vertices, liver_model.surface_mesh.faces, function_smoothed[:, 0])
    #for i in range(20):
    #    plot_mesh_function(liver_model.surface_mesh.vertices, liver_model.surface_mesh.faces, smoother.eigen_vectors[:, i])


if __name__ == '__main__':
    main()