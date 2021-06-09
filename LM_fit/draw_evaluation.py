import igl
import trimesh
import numpy as np
import pymesh
import sys
import os
import copy
import matplotlib.pyplot as plt
from mayavi import mlab
from functools import reduce
from scipy import sparse
import h5py
from PIL import Image, ImageFont, ImageDraw
from util_package.util import tetra_interpolation_full, load_disp_solutions, load_output_xyz
from scipy.spatial.transform import Rotation
import multiprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from util_package.util import load_PC, parse, angle2rotmatrix, angle2drotmatrix, init_reg, load_disp_solutions,\
    parse_non_rigid, normalize, load_voxel_mesh, load_xyz, closest
from util_package.plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label,\
    put_queue, plot_mesh_points_label_anim, update_scene_objs_anim, plot_meshes_n_points, plot_meshes

from Liver import LiverModel, Sampler, ProbeProvider, LaplacianSmoothing, Iterative_Closest_Point
num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

def apply_transform_non_rigid(S, T, R, C, tet_v, templates):
    #apply deformation
    deformation = np.reshape(np.matmul(templates, C), [-1, 3])
    tet_v = tet_v + deformation
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr, deformation


def save_xyz_file(filename, displacement):
    f_org = open('../../org/LiverVolume.nod', 'r')
    f_out = open(filename, 'w')
    for i in range(29545):
        half = f_org.readline()
        half2 = '{:f}'.format(displacement[i, 0]) + ' ' + '{:f}'.format(displacement[i, 1]) + ' ' + '{:f}'.format(displacement[i, 2])
        f_out.write(half[:-1] + ' ' + half2 + '\n')
    f_org.close()
    f_out.close()



def re_order_templates():
    templates = load_disp_solutions('../displacement_solutions/Y1000000.0_P0.49_14p', 42)
    templates = np.reshape(templates, [-1, 3, 42])
    with h5py.File('../V_order.h5', 'r') as hf:
        order = hf['order'][:]
    #templates = templates[order, :, :]
    templates = np.reshape(templates, [-1, 42])
    return templates

def plot_mesh_vectors(i, pc, mesh_v, mesh_f, points, displacemennt, angle=0):
    npoints = points + displacemennt
    mlab.figure(bgcolor=(1,1,1))
    mlab.quiver3d(points[:, 1], points[:, 2], points[:, 0], displacemennt[:, 1], displacemennt[:, 2], displacemennt[:, 0],
                  color=(1, 0.3, 0.3), mode='2ddash', line_width=3, scale_factor=1)
    mlab.triangular_mesh([vert[1] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         [vert[0] for vert in mesh_v],
                         mesh_f,
                         opacity=0.3,
                         color=(0.5, 0.5, 0.5))
    mlab.points3d(points[:, 1], points[:, 2], points[:, 0], scale_factor=0.006, color=(1, 0.1, 0.1))
    mlab.points3d(npoints[:, 1], npoints[:, 2], npoints[:, 0], scale_factor=0.006, color=(0.1, 1, 1))
    mlab.points3d(pc[:, 1], pc[:, 2], pc[:, 0], scale_factor=0.0025, color=(0, 0.4, 0))
    if i == 44:
        mlab.view(azimuth=100+angle, elevation=70.0)
    elif i == 57:
        mlab.view(azimuth=80+angle, elevation=130.0)
    elif i == 67:
        mlab.roll(-10)
        mlab.view(azimuth=360-35+angle, elevation=80.0)
    else:
        mlab.roll(10)
        mlab.view(azimuth=260+angle, elevation=90.0)
    #mlab.view(azimuth=90, elevation=90.0)
    #mlab.show()
    mlab.savefig('tmp.png')
    mlab.close()
    mlab.clf()
    return np.array(Image.open('tmp.png'))

baseline_nonrigid_dir = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/records/reg_out_rigid'
baseline_rigid_dir = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/records/reg_out_nonrigid'
our_nonrigid_dir1 = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/records/our_rigid_2e-3'
our_nonrigid_dir2 = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/records/our_nonrigid_2e-3'
target_dir = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/AdditionalDataForParticipants'

def load_result_compute_distance(i, liver_model):
    dirs = [baseline_nonrigid_dir, baseline_rigid_dir, our_nonrigid_dir1, our_nonrigid_dir2]
    dirs_plot = [baseline_nonrigid_dir, baseline_rigid_dir, our_nonrigid_dir1]
    differences = []
    imgs = []
    for output_dir in dirs:
        displacement = load_output_xyz(output_dir + '/ResultsSet' + str(i).zfill(3) + '.xyz')
        tran_v_non = liver_model.tetr_mesh.vertices + displacement
        targets_org = (1 / 1000) * load_xyz(target_dir + '/LiverTargetsIncomplete.xyz')
        _, closet_vert_target = closest(targets_org, liver_model.vertices)

        GT_target = load_xyz(target_dir + '/Set' + str(i).zfill(3) + '_TargetsIncomplete.xyz')
        GT_target = (1 / 1000) * GT_target

        output_target = tetra_interpolation_full(targets_org, liver_model.vertices, tran_v_non)
        difference = GT_target - output_target
        difference_m = 1000*np.sqrt(np.sum(difference * difference, axis=1))
        differences.append(difference_m)

        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        point_cloud = point_cloud / 1000
        img = plot_mesh_vectors(i, point_cloud, tran_v_non, liver_model.faces, output_target, difference)
        imgs.append(img)

    plt.clf()
    plt.xticks(fontsize=22)
    plt.yticks(np.arange(0, 18, step=2), fontsize=22)
    plt.ylim(0, 18)
    plt.boxplot(differences,
                labels=['Rigid', 'Nonrigid', 'Our1', 'Our2'])
    plt.ylabel('TRE (mm)', fontsize=22)
    plt.xlabel('data #' + str(i), fontsize=22)
    #plt.show()
    plt.tight_layout()
    plt.savefig('./imgs/box_tmp'+str(i)+'.png')

    image = np.concatenate([np.concatenate(imgs[:2], axis=1),
                            np.concatenate(imgs[2:], axis=1)], axis=0)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Roboto-Bold.ttf', size=35)
    color = 'rgb(0, 0, 0)'
    hight =60
    width = text_w("Rigid")
    draw.text((380-width,299-hight), "Rigid", fill=color, align='left', font=font)
    width = text_w("Nonrigid")
    draw.text((780-width,299-hight), "Nonrigid", fill=color, align='left', font=font)
    width = text_w("Our1")
    draw.text((380-width,599-hight), "Our1", fill=color, align='left', font=font)
    width = text_w("Our2")
    draw.text((780-width,599-hight), "Our2", fill=color, align='left', font=font)
    image.save('./imgs/fig'+str(i)+'.png')
    return

def load_result_compute_distance_video(i, liver_model, angle):
    if os.path.isfile('./imgs/fig_' + str(i) + '/' + str(angle) + '.png'):
        return
    dirs = [baseline_nonrigid_dir, baseline_rigid_dir, our_nonrigid_dir1, our_nonrigid_dir2]
    dirs_plot = [baseline_nonrigid_dir, baseline_rigid_dir, our_nonrigid_dir1]
    differences = []
    imgs = []
    for output_dir in dirs:
        displacement = load_output_xyz(output_dir + '/ResultsSet' + str(i).zfill(3) + '.xyz')
        tran_v_non = liver_model.tetr_mesh.vertices + displacement
        targets_org = (1 / 1000) * load_xyz(target_dir + '/LiverTargetsIncomplete.xyz')
        _, closet_vert_target = closest(targets_org, liver_model.vertices)

        GT_target = load_xyz(target_dir + '/Set' + str(i).zfill(3) + '_TargetsIncomplete.xyz')
        GT_target = (1 / 1000) * GT_target

        output_target = tetra_interpolation_full(targets_org, liver_model.vertices, tran_v_non)
        difference = GT_target - output_target
        difference_m = 1000 * np.sqrt(np.sum(difference * difference, axis=1))
        differences.append(difference_m)

        point_cloud, PC_label = load_PC('../../org/reg_dataset/Set' + str(i).zfill(3))
        point_cloud = point_cloud / 1000
        img = plot_mesh_vectors(i, point_cloud, tran_v_non, liver_model.faces, output_target, difference, angle)
        imgs.append(img)

    plt.clf()
    plt.xticks(fontsize=22)
    plt.yticks(np.arange(0, 18, step=2), fontsize=22)
    plt.ylim(0, 18)
    plt.boxplot(differences,
                labels=['Rigid', 'Nonrigid', 'Our1', 'Our2'])
    plt.ylabel('TRE (mm)', fontsize=22)
    plt.xlabel('data #' + str(i), fontsize=22)
    # plt.show()
    plt.tight_layout()
    plt.savefig('./imgs/box_tmp' + str(i) + '.png')

    image = np.concatenate([np.concatenate(imgs[:2], axis=1),
                            np.concatenate(imgs[2:], axis=1)], axis=0)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Roboto-Bold.ttf', size=35)
    color = 'rgb(0, 0, 0)'
    hight = 60
    width = text_w("Rigid")
    draw.text((380 - width, 299 - hight), "Rigid", fill=color, align='left', font=font)
    width = text_w("Nonrigid")
    draw.text((780 - width, 299 - hight), "Nonrigid", fill=color, align='left', font=font)
    width = text_w("Our1")
    draw.text((380 - width, 599 - hight), "Our1", fill=color, align='left', font=font)
    width = text_w("Our2")
    draw.text((780 - width, 599 - hight), "Our2", fill=color, align='left', font=font)
    image.save('./imgs/fig_' + str(i) + '/' + str(angle) + '.png')
    return

def text_w(text):
    image = Image.fromarray(255*np.ones(shape=[800, 800], dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Roboto-Bold.ttf', size=35)
    color = 'rgb(0, 0, 0)'
    draw.text((100,100), text, fill=color, align='left', font=font)
    head_array = np.array(image)
    idicator = np.sum(head_array, axis=0)
    num_empty = np.sum(idicator == np.max(idicator))
    return 800-num_empty

def eval_44_57_67_84():
    liver_model = LiverModel()
    templates = re_order_templates()
    target_dir = '/media/mjia/Data2nd/Research/NonRigid_Registration/Liver_Phantom_data/org/AdditionalDataForParticipants'
    meshes = [pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 2)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 3)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 4)[0], :]),
              pymesh.form_mesh(liver_model.tetr_mesh.vertices,
                               liver_model.tetr_mesh.faces[np.where(liver_model.face_label == 1)[0], :])]
    for i in [44]:#[44, 67, 57, 84]:
        #load_result_compute_distance(i, liver_model)
        for angle in range(36*2):
            load_result_compute_distance_video(i, liver_model, 5*angle)

    img44 = np.array(Image.open('./imgs/box_tmp'+str(44)+'.png'))
    img57 = np.array(Image.open('./imgs/box_tmp'+str(57)+'.png'))
    img67 = np.array(Image.open('./imgs/box_tmp'+str(67)+'.png'))
    img94 = np.array(Image.open('./imgs/box_tmp'+str(84)+'.png'))
    image = np.concatenate([np.concatenate([img44, img57], axis=1),
                            np.concatenate([img67, img94], axis=1)], axis=0)
    image = Image.fromarray(image)
    image.save('./imgs/boxplot.png')


    print('done')

import cv2
def video_44_57_67_84():
    for i in [44, 67, 57, 84]:
        #load_result_compute_distance(i, liver_model)
        out = cv2.VideoWriter('./imgs/fig_' + str(i) + '/video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (800, 600))
        for angle in range(36*2):
            img = cv2.imread('./imgs/fig_' + str(i) + '/' + str(5*angle) + '.png')
            out.write(img)
        cv2.destroyAllWindows()
        out.release()

    img44 = np.array(Image.open('./imgs/box_tmp'+str(44)+'.png'))
    img57 = np.array(Image.open('./imgs/box_tmp'+str(57)+'.png'))
    img67 = np.array(Image.open('./imgs/box_tmp'+str(67)+'.png'))
    img94 = np.array(Image.open('./imgs/box_tmp'+str(84)+'.png'))
    image = np.concatenate([np.concatenate([img44, img57], axis=1),
                            np.concatenate([img67, img94], axis=1)], axis=0)
    image = Image.fromarray(image)
    image.save('./imgs/boxplot.png')



#plot_reg()
def crop_video():
    cap = cv2.VideoCapture('/home/mjia/Videos/ee.mp4')
    out1 = cv2.VideoWriter('implicit_small1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(1890/2), int(1940/2)))
    out2 = cv2.VideoWriter('implicit_small2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(1890/2), int(1940/2)))
    out3 = cv2.VideoWriter('implicit_small3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(1890/2), int(1940/2)))
    n = 0
    while(cap.isOpened()):
        n+=1
        print(n)
        ret, frame = cap.read()
        if n>=11 and n<=277:
            new_frame = frame[220:, 1000:3840-950]
            new_frame = cv2.resize(new_frame, (int(1890/2), int(1940/2)), interpolation=cv2.INTER_LINEAR)
            out1.write(new_frame)
        if n>=276 and n<=542:
            new_frame = frame[220:, 1000:3840-950]
            new_frame = cv2.resize(new_frame, (int(1890/2), int(1940/2)), interpolation=cv2.INTER_LINEAR)
            out2.write(new_frame)
        if n>=537 and n<=803:
            new_frame = frame[220:, 1000:3840-950]
            new_frame = cv2.resize(new_frame, (int(1890/2), int(1940/2)), interpolation=cv2.INTER_LINEAR)
            out3.write(new_frame)
        if n>813:
            break
    cap.release()
    out1.release()
    out2.release()
    out3.release()

#eval_44_57_67_84()

crop_video()