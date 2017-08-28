#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:38:22 2017

@author: xifan
"""

from skimage.segmentation import felzenszwalb
from skimage.data import coffee
from skimage import io
from scipy.misc import imread, imsave, imresize
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

if __name__ == '__main__':
    np.set_printoptions(np.NaN)
    
    root = sys.argv[1] # e.g.'/home/mi/RelationalReasoning/CLEVR_seg/images/
    split = sys.argv[2] # 'train' or 'valid'
    max_num_layers = 0
    
    COORS = np.zeros((320, 480, 2))
    for i in range(320):
        for j in range(480):
            COORS[i, j, 0] = (i-159.5)/159.5
            COORS[i, j, 1] = (j-239.5)/239.5

    if split=='train':
        root_dir = os.path.join(root, 'train')
    #    jpg_dir = os.path.join(root, 'train')
        mat_dir = os.path.join(root, 'train_mat')
        apps_dir = os.path.join(root, 'train_apps_single')
    if split=='valid':
        root_dir = os.path.join(root, 'val')
    #    jpg_dir = os.path.join(root, 'train')
        mat_dir = os.path.join(root, 'val_mat')
        apps_dir = os.path.join(root, 'val_apps_single')
    
    try:
        os.mkdir(mat_dir)
    except:
        pass
    try:
        os.mkdir(apps_dir)
    except:
        pass
        
#    root_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/val'
#    jpg_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/val/'
#    mat_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/val_mat/'
#    apps_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train_apps_single'
    
    img_list = os.listdir(root_dir)
    print (len(img_list))
    for i in range(0, len(img_list)):
        path = os.path.join(root_dir, img_list[i])
        if os.path.isfile(path):
    #        print (path)
    #        print (path[0:len(path)-3]+'jpg')
            if path[len(path)-3:len(path)]=='png' :
                print ('%d/%d' % (i, len(img_list)))
                filename_raw = img_list[i][:len(img_list[i])-4]
#                try:
#                    mat = sio.loadmat(mat_dir + '/' + filename_raw + '.mat')
#                except:
                img = io.imread(path)
                segments = felzenszwalb(img, scale=600.0, sigma=0.5, min_size=500)
                if (np.max(segments) + 1) > max_num_layers:
                    max_num_layers = np.max(segments) + 1
                    
#                    image_marked = np.zeros_like(img)
#                    for idx in range(np.max(segments) + 1):
#                        image_marked[segments==idx, 0] = np.round(random.random() * 255)
#                        image_marked[segments==idx, 1] = np.round(random.random() * 255)
#                        image_marked[segments==idx, 2] = np.round(random.random() * 255)
#                        image_marked[segments==idx, 3] = 255
                
#                print (filename_raw)
                dir_path = os.path.join(apps_dir, filename_raw)
                try:
                    os.mkdir(dir_path)
                except:
                    pass
                
                num_layers = np.max(segments) + 1
                layers = np.zeros((num_layers, 320, 480))
                for l in range(num_layers):
                    layers[l, segments==l] = 1

                layers_resized = np.zeros((num_layers, 32, 32))
                sizes = np.zeros((num_layers-1))
                for l in range(num_layers):
                    
                    if l==0:
                        continue
                    apps = np.zeros((320, 480, 3))
                    app_filename = filename_raw + '_' + str(l) + '.png'
                    apps_path = os.path.join(dir_path, app_filename)
                    
                    selection = np.expand_dims(layers[l], -1)
                    selection = np.tile(selection, [1, 1, 4])
                    apps = img * selection
                    min_x = min((np.argwhere(layers[l]==1))[:, 0])
                    max_x = max((np.argwhere(layers[l]==1))[:, 0])
                    min_y = min((np.argwhere(layers[l]==1))[:, 1])
                    max_y = max((np.argwhere(layers[l]==1))[:, 1])

                    crop = apps[min_x:max_x, min_y:max_y]
                    len_x = max_x - min_x + 1
                    len_y = max_y - min_y + 1
                    length = max(len_x, len_y)
                    if len_x > len_y:
                        pad = int((len_x - len_y)/2)
                        crop = np.lib.pad(crop, ((0, 0), (pad, len_x-len_y-pad), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    if len_y > len_x:
                        pad = int((len_y - len_x)/2)
                        crop = np.lib.pad(crop, ((pad, len_y-len_x-pad), (0, 0), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    crop = imresize(crop, (128, 128))
                    imsave(apps_path, crop)
#                    print (apps_path)
                    sizes[l-1] = float(length/128)
                    
#                    for l in range(num_layers):
#                        for i in range(128):
#                                bl = np.mean(layers[l, i*10:(i + 1)*10, j*15:(j + 1)*15])
#    #                            bl = np.ceil(bl) # 0/1
#                                layers_resized[l, i, j] = bl
                coors = COORS
                coors = np.expand_dims(coors, axis=0)
                coors = coors * np.expand_dims(layers, axis=-1)
                coors = np.mean(coors, axis=1)
                coors = np.mean(coors, axis=1)
                
#                layers_resized = np.ceil(layers_resized)
#                print (layers_resized)
            
#                io.imsave(jpg_dir + filename_raw +'.jpg', image_marked)
                sio.savemat(mat_dir + '/' + filename_raw + '.mat', {'masks':layers, 'num_layers':num_layers, 'coors':coors, 'sizes':sizes})