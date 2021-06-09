import igl
import trimesh

v, f = igl.read_triangle_mesh('../org/Liver.off')
k = igl.gaussian_curvature(v, f)

mesh_liver = trimesh.Trimesh(vertices=v, faces=f)
mesh_liver.show()