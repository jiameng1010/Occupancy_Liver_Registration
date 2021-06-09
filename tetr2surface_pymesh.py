import pymesh
mesh = pymesh.load_mesh('../org/Liver.off')
mesh, _ = pymesh.remove_isolated_vertices(mesh)
pymesh.save_mesh('../org/Liver_surface.off', mesh)
print(' ')
