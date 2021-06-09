from mayavi import mlab
from util import load_PC
import numpy as np
import multiprocessing

def print_square(points):
    for i in range(100):
        pp = np.random.normal(0.0, 10, size=pp.shape)
        points.put(3)

def calc_sq(num, q):
    for n in num:
        q.put(n*n)

def main():
    num = [2, 3, 4]

    p = multiprocessing.Queue()
    q = multiprocessing.Process(target=print_square, args=p)
    q.start()
    q.join()

main()