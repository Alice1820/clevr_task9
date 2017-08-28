import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from transforms import Scale
import os
from scipy.misc import imread, imsave, imresize
import scipy.io as sio

MX_N = 23

#train_root_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train'
#train_mat_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train_mat_full_single'
#train_apps_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train_apps_single'
#
#train_root_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train'
#train_mat_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train_mat_full_single'
#train_apps_dir = '/home/mi/RelationalReasoning/CLEVR_seg/images/train_apps_single'

class CLEVR(Dataset):
    def __init__(self, root, segs_root, split='train'):
        with open('data/' + split + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
            
        self.root_dir = root
        self.mat_dir = segs_root + '/' + split + '_mat' # e.g.'/home/mi/RelationalReasoning/CLEVR_seg/images/train_mat_full_single'
        self.apps_dir = segs_root + '/' + split + '_apps_single' # e.g.'/home/mi/RelationalReasoning/CLEVR_seg/images/train_apps_single'
            
        self.data
        self.transform0 = transforms.Compose([
                                Scale([128, 128]),
                                transforms.Pad(4),
                                transforms.RandomCrop([128, 128]),
                                transforms.ToTensor(),
                                ])
    
        self.transform1 = transforms.Compose([
                                Scale([128, 128]),
                                transforms.Pad(4),
                                transforms.RandomCrop([128, 128]),
                            #    transforms.ToTensor(),
                            #    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            #                        std=[0.5, 0.5, 0.5])
                                ])
                            
        self.transform2 = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])
        self.if_aug = (split=='train')
        
        self.transform_app = transforms.Compose([
#                                Scale([128, 128]),
                                transforms.ToTensor(),
                                ])
        self.split = split

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
#        print (imgfile)
        dir_path = os.path.join(self.apps_dir, imgfile[0:len(imgfile)-4])
        mat_path = os.path.join(self.mat_dir, imgfile[0:len(imgfile)-4] + '.mat')
        mat = sio.loadmat(mat_path)
        num_layers = mat['num_layers'] - 1 # no background
        
#        print (int(num_layers))
        
#        img = Image.open(os.path.join(self.root, 'images', self.split, imgfile)).convert('RGB')
        
        masks = np.zeros((MX_N, 32, 32))
        masks_part = mat['masks']
        masks[:int(num_layers)] = masks_part[1:]
        
#        print (masks_part)
        apps = np.zeros((MX_N, 128, 128, 3), dtype=np.uint8)
        for l in range(1, int(num_layers)+1):
            app_path = os.path.join(dir_path, imgfile[0:len(imgfile)-4] + '_' + str(l) + '.png')
            app_img = imread(app_path)
#            app_img = Image.open(app_path).convert('RGB')
#            app_img = self.transform_app(app_img)
#            apps.append(app_img)
            app_img = imresize(app_img, [128, 128])
            apps[l-1] = app_img[:, :, 0:3]
#            print (app_path)
            
##            imshow
#            import matplotlib.pyplot as plt
#            fig = plt.figure()
#            ax = fig.add_subplot(111)
#            ax.imshow(app_img)
#            plt.show()
                
        
#        print (apps)
        
        coors = np.zeros((MX_N, 2))
        coors_part = mat['coors']
        coors[:int(num_layers)] = coors_part[1:int(num_layers)+1]
#        sizes = np.zeros((MX_N))
#        sizes_part = mat['sizes']
#        sizes[:int(num_layers)] = sizes_part[:int(num_layers)]
#        apps_tensor = torch.FloatTensor(apps.asdtype(float))
        
#        print (img)
        
        if self.if_aug:
#            img = self.transform1(img)
#            angle = random.random()*2.8648*2 - 2.8648 # -0.05-0.05
#            img = img.rotate(angle, resample=Image.BILINEAR)
##            print (img)
#            img = self.transform2(img)
            
            apps_tensor = []
            for l in range(int(num_layers)):
#                print (apps[l], 'apps[l]')
                transform_tmp = transforms.Compose([
                                transforms.ToPILImage(),
                                Scale([32, 32]),
                                transforms.Pad(1),
                                transforms.RandomCrop([32, 32]),
                                ])
                apps_tmp = transform_tmp(apps[l])
                
                angle = random.random()*2.8648*2 - 2.8648 # -0.05-0.05
                apps_tmp = apps_tmp.rotate(angle, resample=Image.BILINEAR)
#                print (apps_tmp, 'apps_tmp') # PIL.Image.Image image mode=RGB size=128x128
                apps_tmp = self.transform2(apps_tmp)
                
#                print (apps_tmp)
                apps_tensor.append(apps_tmp)
                
            apps_tensor = torch.stack(apps_tensor)
                
        else:
#            img = self.transform0(img)
            apps_tensor = []
            for l in range(int(num_layers)):
#                print (apps[l], 'apps[l]')
                transform_tmp = transforms.Compose([
                                transforms.ToPILImage(),
                                Scale([32, 32]),
                                transforms.ToTensor(),
                                ])
                apps_tmp = transform_tmp(apps[l])
                apps_tensor.append(apps_tmp)
            apps_tensor = torch.stack(apps_tensor)
        
#        print (apps_tensor)
        apps_tensor_pad = torch.FloatTensor(np.zeros((MX_N, 3, 32, 32)))
        apps_tensor_pad[:int(num_layers)].copy_(apps_tensor)
        
#        print (apps_tensor_pad) # 23*3*32*32
#        print (img) # 3*128*128
        
        coors = torch.FloatTensor(coors)
        masks_tensor_pad = torch.FloatTensor(masks)
#        sizes = torch.FloatTensor(sizes)
        
        return apps_tensor_pad, masks_tensor_pad, int(num_layers), question, len(question), answer, family

    def __len__(self):
        return len(self.data)

#transform1 = 
#transform2 = transforms.Compose([
#    Scale([128, 128]),
#    transforms.Pad(4),
#    transforms.RandomCrop([128, 128]),
##    transforms.ToTensor(),
##    transforms.Normalize(mean=[0.5, 0.5, 0.5],
##                        std=[0.5, 0.5, 0.5])
#])


def collate_data(batch):
    
    appss, maskss, nums_layers, questions, lengths, answers, families = [], [], [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[3]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[3]), reverse=True)

    for i, b in enumerate(sort_by_len):
        apps, masks, num_layers, question, length, answer, family = b
        length = len(question)
        questions[i, :length] = question
        appss.append(apps)
        maskss.append(masks)
        nums_layers.append(num_layers)
#        coorss.append(coors)
#        sizess.append(sizes)
        lengths.append(length)
        answers.append(answer)
        families.append(family)

    return torch.stack(appss), torch.stack(maskss), nums_layers, torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), families
