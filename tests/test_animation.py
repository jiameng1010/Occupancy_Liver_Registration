from mayavi import mlab
from util import load_PC
import numpy as np
import h5py
import multiprocessing
import time

def print_square(num, points):
    pp, _ = load_PC('../../org/datasets/Set001')
    for i in range(20):
        pp = pp + np.random.normal(0.0, 10, size=pp.shape)
        points.put(pp)
        time.sleep(1)

def main():
    points, _ = load_PC('../../org/datasets/Set001')
    mlab.figure()
    num=[1]

    p = multiprocessing.Queue()
    q = multiprocessing.Process(target=print_square, args=(num, p))
    result = anim(points, p)
    q.start()
    #q.join()
    mlab.show()

def updates(f, points):
    f.mlab_source.x = points[:, 0]
    f.mlab_source.y = points[:, 2]
    f.mlab_source.z = points[:, 1]
    #f.scene.camera.azimuth(10)
    #f.scene.render()

@mlab.animate(delay=200, ui=False)
def anim(points, p):
    '''while points.empty is False:
        p = points.get()
        f = mlab.points3d(p[:,0], p[:,2], p[:,1])
    #mlab.show()'''
    f = mlab.points3d(points[:, 0], points[:, 2], points[:, 1])
    while 1:
        try:
            pp = p.get()
        except:
            continue
        #points = points+np.random.normal(0.0, 10, size=points.shape)
        updates(f, pp)
        yield
    mlab.close()

main()
main()

#anim() # Starts the animation.
#mlab.show()